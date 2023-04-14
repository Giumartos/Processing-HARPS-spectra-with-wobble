'''
Esse programa utiliza a biblioteca wobble para: 

- Fazer um processamento de espectros do HARPS de uma estrela tomados em diferentes datas de observação

- Reconhecer as linhas telúricas e separar qual é o espectro da estrela e qual é o espectro telúrico

- Obter velocidade radial da estrela (um valor para cada observação)

Posteriormente:

- Manipulo arquivos no formato .hdf5 para acessar os resultados obtidos

- Calculo a razão S/N do espectro gerado para a estrela

- Salvo o espectro e informações importantes no formato .fits
'''

# importanto as bibliotecas necessárias
import wobble
import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt
import glob
import os
import h5py
from astropy.io import fits
from tqdm import tqdm
import ctypes # integracao com C
np.random.seed(0)

# para tirar avisos do tensorflow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)

# para plots
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['figure.figsize'] = (15.0, 5.0)

def c_init(c_name):
    # Funtion to initialize the C shared library
    c_library = ctypes.CDLL(c_name)

    # Defining functions argument types
    # types das variaveis da minha função em C

    c_library.where.argtypes = [ctypes.c_long,
                                ctypes.c_float]

    c_library.where.restype = None # type da variavel que é retornada

    return c_library

def save_spec_csv(out_file_name, data, SNR, mean_rv, rv_err, star_name):
    '''
    Escreve arquivo com espectro no formato .txt

    out_file_name: nome do arquivo de saida
    data: espectro no formato de matriz com comp de onda na coluna [0] e fluxos na coluna [1]
    SNR: razao sinal/ruido do espectro
    rv: velocidade radial media
    star_name: nome da estrela
    '''
    print("\n*** Saving CSV file ***\n")

    # remove o arquivo anterior se já existia
    if os.path.exists(out_file_name):
        os.remove(out_file_name)

    head = "Star: {}\nSNR = {:.2f}\nMean rv: {:.2f} +/- {:.2f}\n".format(star_name, SNR, mean_rv, rv_err)

    np.savetxt(out_file_name, data, header = head, delimiter=',', fmt = '%f')


def save_spec_fits(filename, spectrum, mean_SNR, mean_rv, rv_err, star_name):
    '''
    Salvar espectro dado como matriz (array) no formato .fits

    spectrum: matriz do espectro com a primeira coluna (spectrum[0]) sendo comp de onda e
    a segunda coluna (spectrum[1]) os fluxos
    filename: nome do arquivo de saida do espectro

    Based on https://github.com/MatheusJCastro/ascii2fits/blob/main/ascii2fits.py
    '''

    print("\n*** Saving .fits file ***\n")

    crval1 = spectrum[0][0]  # get the first wavelength value
    cdelt1 = float(spectrum[0][1] - crval1)  # calculate the wavelength step

    hdu = fits.PrimaryHDU(spectrum[1])  # create the Primary HDU of the fits with flux values
    hdu.header["CRVAL1"] = crval1  # add to the header the first wavelength value
    hdu.header["CDELT1"] = float("{:.5f}".format(cdelt1))  # add to the header the wavelength step
    hdu.header["STAR"] = star_name
    hdu.header["MEAN SNR"] = float("{:.2f}".format(mean_SNR))
    hdu.header["MEAN RV"] = float("{:.2f}".format(mean_rv))
    hdu.header["RV UNCERTAINTY"] = float("{:.2f}".format(rv_err))

    hdul = fits.HDUList([hdu])
    hdul.writeto(str(filename) + ".fits", overwrite=True)  # save the spectrum

def get_mean_SNR(flux, step):
    '''
    Dada uma array de fluxos, calcula a razão S/R média (calcula o S/R em pequenas partes
    e depois faz a média, pois o valor do sinal/ruído depende da região analisada).
    Para isso, chama o programa DER_SNR várias vezes

    step: tamanho do pedaço do espectro para calcular o SNR
    '''
    # importando a funcao de calcular SNR de outro programa
    import DER_SNR as snr

    # calculando o SNR em diferentes pedaços do espectro

    SNRs = []  # valores do SNR nos pedaços

    i_ant = 0
    while i_ant < len(flux):
        if (i_ant + step) < len(flux):
            i = i_ant + step  # tamanho do pedaço do espectro
        else:
            break # se o pedaço cair fora da array

        sig = snr.DER_SNR(flux[i_ant:i])

        if np.isfinite(sig):
            SNRs.append(sig)

        i_ant = i

    # para o pedaço que sobrou, se tiver tamanho menor que step(nao entrou no while anterior)
    SNRs.append(snr.DER_SNR(flux[i_ant:]))
    result = np.mean(SNRs)

    return result

def get_mean_rv(rvs_filename):
    """
    Dado o nome do arquivo em que foram salvas as velocidades radiais,
    retorna seu valor medio e o erro medio
    """
    data = np.loadtxt(rvs_filename, skiprows=4, delimiter=' ', dtype="str")
    rvs = data[:,1].astype(float)
    rvs_err = data[:, 2].astype(float)

    return np.mean(rvs), np.mean(rvs_err)

def plot_spectrum(x, y, figname, star_name, tellurics=False):

    plt.figure(figsize=(16, 8))
    plt.plot(x, y, c='b')
    if tellurics:
        title = 'Tellurics - ' + star_name
    else:
        title = star_name
    plt.title(title)
    plt.tight_layout()
    plt.savefig(figname)
    plt.show()


def pre_process_data (star_name, data_dir, hdf5_file_processed_data):
    '''
    Essa função faz o pre-processamento dos espectros do HARPS usando o wobble

    star_name: nome da estrela
    data_dir: nome da pasta onde estao os espectros 2D (.e2ds) e arquivos de ccf
    hdf5_file_processed_data: nome do arquivo .hdf5 para salvar dados pre-processados
    
    Resulta em um arquivo no formato .hdf5 com os dados das observacoes/espectros das estrelas
    e os espectros pre-processados
    '''
    # objeto do wobble para guardar os dados
    data = wobble.Data()

    filenames = glob.glob(data_dir + '/HARPS*_ccf_*')

    # se nao encontra os arquivos, para a execucao
    if len(filenames) == 0:
        print("Files not found")
        exit()

    # obtem os espectros da estrela, usando a função from_HARPS
    for filename in tqdm(filenames):
        try:
            sp = wobble.Spectrum()
            sp.from_HARPS(filename, process=True)
            data.append(sp)
        except Exception as e:
            print("File {0} failed; error: {1}".format(filename, e))
    
    # tirando pontos com SNR < 10
    data.drop_bad_orders(min_snr=10.)
    data.drop_bad_epochs(min_snr=10.)

    # salvando os espectros carregados e processados em um arquivo .hdf5
    # remove o arquivo anterior se já existia, senão aparece warning
    if os.path.exists(hdf5_file_processed_data):
        os.remove(hdf5_file_processed_data)

    # escreve os dados da das observações da e os espectros pre processados no arquivo .hdf5    
    data.write(hdf5_file_processed_data)


def get_real_spectra(hdf5_file_processed_data, hdf5_file_results, rvs_filename):
    '''
    Vai usar o wobble para identificar as linhas telúricas e separar o espectro
    telúrico do espectro da estrela

    hdf5_file_processed_data: arquivo .hdf5 gerado pelo tratamento feito na função "pre_process_data"
    hdf5_file_results: nome do arquivo .hdf5 para salvar os resultados
    rvs_filename: nome do arquivo txt para salvar as velocidades radiais

    Resulta em um arquivo .hdf5 com dados e os espectros "real" (sem interferencias teluricas) da estrela e o
    espectro telurico
    '''
    # objeto wobble com os dados
    data = wobble.Data(filename=hdf5_file_processed_data)

    # objeto wobble para os resultados
    results = wobble.Results(data=data)

    # fazendo as correções
    print("\n*** Starting telluric corrections ***\n")

    for r in range(len(data.orders)):
        print('starting order {0} of {1}'.format(r+1, len(data.orders)))
        model = wobble.Model(data, results, r)
        model.add_star('star')
        model.add_telluric('tellurics', variable_bases=2) # variable tellurics
        wobble.optimize_order(model)

    # obtendo velocidades radiais da estrela
    print("\n*** Obtaining radial velocities ***\n")
    results.combine_orders('star') # combinando ordens

    # Corrigindo das velocidades instrumentais e baricêntrica:
    results.apply_drifts('star') # instrumental drift corrections
    results.apply_bervs('star') # barycentric corrections

    # salvando as velocidades radiais
    if os.path.exists(rvs_filename):
        os.remove(rvs_filename)

    results.write_rvs('star', rvs_filename)

    # Escrevendo resultados da correção telúrica em um arquivo no formato .hdf5:
    print("\n*** Saving results ***\n")
    if os.path.exists(hdf5_file_results):
        os.remove(hdf5_file_results)

    results.write(hdf5_file_results)


def correct_superposition_orders(waves, fluxes):
    """
    Dadas arrays com comp de onda e fluxos com as sobreposicoes
    das ordens, retorna array sem os dados sobrepostos

    waves: array de comprimentos de onda
    fluxes: array de fluxos
    """
    epsilon = 0.001 # intervalo para considerar mesmo valor

    # usa funcao que fiz em C para achar as posicoes repetidas. Vai gerar um arquivo chamado "reps_matrix.txt"
    c_lib.where(len(waves), epsilon)

    # lendo a matrix do arquivo de repeticoes e ja excluindo ele
    # nessa matriz, a linha i corresponde ao elemento i da array com repeticoes, e os elementos da linha i sao os indices
    # onde ha repeticao
    reps_matrix = np.loadtxt("reps_matrix.txt", dtype='int', unpack=False)
    os.remove("reps_matrix.txt")

    # para os fluxos, a primeira ocorrencia tera o valor substituito pela mediana dos dos fluxos das outras
    # ocorrencias do comp de onda

    for l in range(len(reps_matrix)): # indices de linha
        idx_rep = np.array([]).astype(np.int64)
        for c in range(len(reps_matrix[0])): # indices de coluna
            if reps_matrix[l][c] != 0:
                idx_rep = np.append(idx_rep, reps_matrix[l][c])
        if len(idx_rep) > 0:
            fluxes[l] = np.median(np.array([fluxes[l], fluxes[idx_rep][0]]))

    # tranformando a matriz de repetições em uma unica array (flat)
    reps_array = np.ndarray.flatten(reps_matrix)

    # removendo os zeros, que sao so da inicializacao da array no C
    zeros = np.array([]).astype(np.int64)
    for i in range(len(reps_array)):
        if reps_array[i] == 0:
            zeros = np.append(zeros, i)
    reps_array = np.delete(reps_array, zeros)

    # tirando as repetições dos fluxos e comp de onda, deixando a primeira ocorrencia
    waves_norep = np.delete(waves, reps_array)
    fluxes_norep = np.delete(fluxes, reps_array)

    return waves_norep, fluxes_norep


def get_waves_fluxes(results_filename_hdf5, waves_filename):

    """
    Dado o nome do arquivo de resultados obtido apos correcoes teluricas,
    retorna comprimento de onda e fluxos da estrela e teluricos

    results_filename_hdf5: arquivo gerado pela funcao "get_real_spectra", com
    os resultados dos espectros apos correcao telurica

    wavelengths_star: comp de onda do espectro corrigido da estrela
    fluxes_star: fluxos do espectro corrigido da estrela
    wavelengths_tellurics: comp de onda do espectro telurico
    fluxes_tellurics: fluxos do espectro telurico

    """
    print("\n*** Obtaining wavelengths and fluxes ***\n")

    # comprimentos de onda e fluxos considerando valores sobrepostos nas ordens
    wavelengths_star, wavelengths_tellurics = np.array([]), np.array([])
    fluxes_star, fluxes_tellurics = np.array([]), np.array([])

    with h5py.File(results_filename_hdf5, 'r') as hdf: #abrindo
    
        R = np.array(hdf.get('R')) # numero de ordens

        # para todas as ordens, vai colocar os comprimentos de onda e fluxos
        # mesmo com as sobreposicoes de comp de onda entre elas (vai remover depois)
        for r in range(R): # 72 para o HARPS
            order = hdf.get('order' + '{}'.format(r))

            waves_star_order = np.exp(np.array(order.get('star_template_xs'))) # comp de onda daquela ordem
            wavelengths_star = np.append(wavelengths_star, waves_star_order) # coloca na array de todos comp onda

            fluxes_star_order = np.exp(np.array(order.get('star_template_ys')))
            fluxes_star = np.append(fluxes_star, fluxes_star_order)
            
            waves_tellurics_order = np.exp(np.array(order.get('tellurics_template_xs')))
            wavelengths_tellurics = np.append(wavelengths_tellurics, waves_tellurics_order)

            fluxes_tellurics_order = np.exp(np.array(order.get('tellurics_template_ys')))
            fluxes_tellurics = np.append(fluxes_tellurics, fluxes_tellurics_order)

    # salvando em um arquivo para abrir mais facil com o C
    np.savetxt(waves_filename, wavelengths_star.T, delimiter='\n', fmt='%4f')

    # obtendo comp de onda e fluxos da estrela, sem as sobreposicoes:
    print("\n*** Removing superpositions star ***")
    wavelengths_star, fluxes_star = correct_superposition_orders(wavelengths_star, fluxes_star)
    #print("\n*** Removing superpositions tellurics ***")
    #wavelengths_tellurics, fluxes_tellurics = correct_superposition_orders(wavelengths_tellurics, fluxes_tellurics)

    print("\n*** Waves and fluxes: OK ***")

    # retornando as arays de comp de onda e fluxos da estrela e telurico
    return wavelengths_star, fluxes_star, wavelengths_tellurics, fluxes_tellurics


def run(star_name, data_dir, results_dir):

    if not os.path.exists(results_dir):  # cria diretorio se nao existe
        os.makedirs(results_dir)

    # nomes dos arquivos para salvar espectros pre-processados (hdf5), resultados(hdf5) e rvs(txt)
    processed_data_filename = results_dir + "processed_data_" + star_name + ".hdf5"
    results_filename = results_dir + "results_" + star_name + ".hdf5"
    waves_filename = results_dir + "waves_with_reps" + ".csv"
    rvs_filename = results_dir + "rvs_" + star_name + ".txt"

    # nomes dos arquivos para salvar os espectros da estrela e telurico
    final_spectrum_star = results_dir + "spectrum_" + star_name
    final_spectrum_tellurics = results_dir + "spectrum_tellurics"

    # pre-processamento dos espectros
    #pre_process_data(star_name, star_name+"/", processed_data_filename)

    # fazendo as correcoes teluricas
    #get_real_spectra(processed_data_filename, results_filename, rvs_filename)

    # obtendo comprimento de onda e fluxos dos espectros da estrela e telurico
    waves_star, fluxes_star, waves_tellurics, fluxes_tellurics = get_waves_fluxes(results_filename, waves_filename)

    exit()
    # obtendo velocidade radial media
    mean_rv, rv_err = get_mean_rv(rvs_filename)

    # obtendo SNR medio do espectro da estrela
    mean_SNR = get_mean_SNR(fluxes_star, 200)

    # matriz com o espectro
    spectrum = np.array([waves_star, fluxes_star])

    # salvando espectros no formato .csv out_file_name, data, SNR, rv, star_name
    save_spec_csv(final_spectrum_star + ".csv", spectrum, mean_SNR, mean_rv, rv_err, star_name)

    # salvando espectro no formato .fits
    save_spec_fits(star_name+ "_n.fits", spectrum, mean_SNR, mean_rv, rv_err, star_name)

    # plotando espectro
    plot_spectrum(waves_star, fluxes_star, results_dir + star_name+".png", star_name, False) # estrela
    plot_spectrum(waves_tellurics, fluxes_tellurics, results_dir + star_name + "_tellurics.png", star_name, True) # telurico

def main():

    star_name = "nome da estrela" # deve ser o nome da pasta com seus espectros

    # diretorio onde estao os espectros da estrela e ccfs
    data_dir = "caminho do diretorio" + star_name

    # diretorio para salvar os dados pre-processados e os resultados
    # results_dir = data_dir + "/Results/"
    results_dir = "caminho do diretorio"

    # rodando as correcoes
    run(star_name, data_dir, results_dir)

if __name__ == '__main__':

    so_file = "caminho do diretorio para o programa em C que encontra repetições na array"
    c_lib = c_init(so_file)  # initialize the C library

    main()
    
