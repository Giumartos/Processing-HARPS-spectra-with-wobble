"""
Esse programa utiliza a biblioteca wobble para: 

- Fazer um processamento de espectros do HARPS de uma estrela tomados em diferentes datas de observação

- Reconhecer as linhas telúricas e separar qual é o espectro da estrela e qual é o espectro telúrico

- Obter velocidade radial da estrela (um valor para cada observação)

Posteriormente:

- Manipulo arquivos no formato .hdf5 para acessar os resultados obtidos

- Calculo a razão S/N do espectro gerado para a estrela

- Salvo o espectro e informações importantes no formato .fits
"""

# importando as bibliotecas necessárias
import wobble
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
import glob
import os
import h5py
from astropy.io import fits
from tqdm import tqdm
import ctypes # integracao com C
import pandas as pd
from astropy.table import Table as tb
from PyAstronomy import pyasl
from astropy import constants as const

# para tirar avisos do tensorflow
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)

# para plots
import matplotlib as mpl
mpl.rcParams['xtick.labelsize'] = 22
mpl.rcParams['ytick.labelsize'] = 22
mpl.rcParams['axes.labelsize'] = 24
mpl.rcParams['figure.figsize'] = (15.0, 5.0)

def save_spec_csv(out_file_name, data):
    """
    Escreve arquivo com espectro no formato .csv

    - out_file_name: nome do arquivo de saída
    - data: espectro no formato de matriz com comp de onda na coluna [0] e fluxos na coluna [1]
    """
    print("\n*** Saving CSV file ***\n")

    # remove o arquivo anterior se já existia
    if os.path.exists(out_file_name):
        os.remove(out_file_name)

    df = pd.DataFrame(data)
    df.to_csv(out_file_name, index=False, header=None)


def save_spec_fits(out_file_name, spectrum, SNR, median_rv, rv_err, star_name):
    """
    Salvar espectro dado como matriz (array) no formato .fits
    - out_file_name: nome do arquivo de saida do espectro
    - spectrum: matriz do espectro com a primeira coluna (spectrum[0]) sendo comp de onda e
    a segunda coluna (spectrum[1]) os fluxos
    - SNR: razão sinal/ruido do espectro
    - median_rv: velocidade radial mediana
    -rv_err: erro da velocidade radial
    - star_name: nome da estrela
    """

    print("\n*** Saving .fits file ***\n")

    if os.path.exists(out_file_name):
        os.remove(out_file_name)

    x_smooth = np.linspace(spectrum[:,0][0], spectrum[:,0][-1],len(spectrum[:,0]))
    obs_flux_interp = np.interp(x_smooth, spectrum[:,0], spectrum[:,1])
    c = {'wave':(x_smooth),'flux':(obs_flux_interp)}
    d = pd.DataFrame(c)
    t = tb.from_pandas(d)

    header = {"STAR  ": star_name, "SNR  ": SNR, "RV  ": median_rv, "RV_ERR  ": rv_err} # coloca essas infos no header do espectro
    pyasl.write1dFitsSpec(out_file_name,t['flux'],t['wave'], header=header)
   

def get_SNR(flux):
    """
    Dada uma array de fluxos, calcula a razão S/N na região de interesse.
    Para isso, chama o programa DER_SNR.
    """
    # importando a funcao de calcular SNR de outro programa
    import DER_SNR as snr

    return snr.DER_SNR(flux)


def get_median_rv(rvs_filename):
    """
    Dado o nome do arquivo em que foram salvas as velocidades radiais,
    retorna seu valor mediano e o erro medio
    """
    data = np.loadtxt(rvs_filename, skiprows=4, delimiter=' ', dtype="str")
    rvs = data[:,1].astype(float)
    rvs_err = data[:, 2].astype(float)

    return np.median(rvs), np.mean(rvs_err)


def plot_spectrum(x, y, figname, star_name, tellurics=False):
    """
    Faz o plot do espectro. Se tellurics=True, plota telúrico.
    Por default, plota espectro da estrela.
    """
    plt.figure(figsize=(16, 8))
    plt.plot(x, y, c='b')
    if tellurics:
        title = 'Tellurics - ' + star_name
    else:
        title = star_name
    plt.title(title, fontsize=26)
    plt.xlabel(r"$\lambda$ ($\AA$)")
    plt.ylabel("Flux")
    plt.tight_layout()
    plt.savefig(figname)
    #plt.show()


def pre_process_data (star_name, data_dir, hdf5_file_processed_data):
    """
    Essa função faz o pre-processamento dos espectros do HARPS usando o wobble

    - star_name: nome da estrela
    - data_dir: nome da pasta onde estao os espectros 2D (.e2ds) e arquivos de ccf
    - hdf5_file_processed_data: nome do arquivo .hdf5 para salvar dados pre-processados
    
    Resulta em um arquivo no formato .hdf5 com os dados das observacoes/espectros das estrelas
    e os espectros pre-processados
    """
    # objeto do wobble para guardar os dados
    data = wobble.Data()
    print(data_dir)
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

    # salvando os espectros carregados e processados em um arquivo .hdf5
    # remove o arquivo anterior se já existia, senão aparece warning
    if os.path.exists(hdf5_file_processed_data):
        os.remove(hdf5_file_processed_data)

    # escreve os dados da das observações dos espectros pre processados no arquivo .hdf5    
    data.write(hdf5_file_processed_data)


def get_real_spectra(hdf5_file_processed_data, hdf5_file_results, rvs_filename):
    """
    Vai usar o wobble para identificar as linhas telúricas e separar o espectro
    telúrico do espectro da estrela

    - hdf5_file_processed_data: arquivo .hdf5 gerado pelo tratamento feito na função "pre_process_data"
    - hdf5_file_results: nome do arquivo .hdf5 para salvar os resultados
    - rvs_filename: nome do arquivo txt para salvar as velocidades radiais

    Resulta em um arquivo .hdf5 com dados e os espectros "real" (sem interferencias teluricas) da estrela e o
    espectro telurico
    """
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

def c_init(c_name):
    """
    Funtion to initialize the C shared library.
    Será usado um programa em C para remover sobreposições de
    comprimento de onda nos espectros.
    """
    c_library = ctypes.CDLL(c_name)

    # Defining functions argument types
    c_library.where.argtypes = [ctypes.c_long, # tipo das variaveis que a função "where" em C vai receber
                                ctypes.c_float,
                                ctypes.c_char_p]

    c_library.where.restype = None # tipo da variavel que é retornada 
    
    return c_library

def python_correct_superposition_orders(waves, fluxes):
    """
    *** Demora muito mais que a função C_correct_superposition_orders***

    Dadas arrays com comp de onda e fluxos com as sobreposicoes
    das ordens, retorna arrays sem os dados sobrepostos

    waves: array de comprimentos de onda
    fluxes: array de fluxos
    """
    # transformei em string para poder comparar apenas 2 casas decimais
    waves_strings = np.array(["%.2f" % x for x in waves])

    i = 0
    while i < len(waves_strings):

        # printando o progresso
        if i % 10 == 0:
            print("Progress {} out of {}.".format(i, len(waves_strings)), end="\r")

        # posicoes com comp de onda repetido
        pos = np.where(waves_strings == waves_strings[i][:len(waves_strings[i]) + 1])[0]

        # removo os comp de onda repetidos, deixando apenas a primeira ocorrencia
        waves_strings = np.delete(waves_strings, pos[1:])

        # para os fluxos, a primeira ocorrencia tera o valor substituito pela
        # mediana dos dos fluxos das outras ocorrencias do comp de onda, e as
        # repeticoes serao removidas
        fluxes[i] = np.median(fluxes[pos][0])
        fluxes = np.delete(fluxes, pos[1:])

        i += 1

    waves_res = waves_strings.astype(np.float)

    return waves_res, fluxes


def C_correct_superposition_orders(waves, fluxes, waves_filename):
    """
    Dadas arrays com comp de onda e fluxos com as sobreposicoes
    das ordens, retorna arrays sem os dados sobrepostos

    waves: array de comprimentos de onda
    fluxes: array de fluxos
    waves_filename: nome do arquivo com os comprimentos de onda para achar repetição
    """
    epsilon = 0.01 # intervalo para considerar mesmo valor 

    print("\nCalling C function to find superpositions in wavelength\n")

    # usa funcao que fiz em C para achar as posicoes repetidas. Vai gerar um arquivo chamado "reps_matrix.txt"
    c_lib.where(len(waves), epsilon, waves_filename.encode("utf-8"))
    
    # lendo a matriz do arquivo de repeticoes e ja excluindo ele
    # nessa matriz, a linha i corresponde ao elemento i da array com repeticoes, e os elementos da linha i sao os indices
    # onde ha repeticao desse elemento i
    print("\nLoading matrix in Python")
    reps_matrix = np.loadtxt("reps_matrix.txt", dtype='int', unpack=False)
    os.remove("reps_matrix.txt")
    
    # para os fluxos, a primeira ocorrencia tera o valor substituido pela mediana dos dos fluxos das outras
    # ocorrencias do comp de onda
    print("\nStarting removing overlapping fluxes")
    for l in range(len(reps_matrix)): # indices de linha
        idx_rep = np.array([]).astype(np.int64)
        for c in range(len(reps_matrix[0])): # indices de coluna
            if reps_matrix[l][c] != 0:
                idx_rep = np.append(idx_rep, reps_matrix[l][c])
            else:
                break
        if len(idx_rep) > 0:
            fluxes[l] = np.median(np.array([fluxes[l], fluxes[idx_rep][0]]))

    # vendo quais foram os indices que elementos repetidos para tirar dos waves e fluxos (não tira primeira ocorrência)
    diff_zero = []
    for i in range(len(reps_matrix)):
        for j in range(len(reps_matrix[0])):
            if reps_matrix[i][j] != 0:
                diff_zero.append(reps_matrix[i][j])

    # tirando os elementos repetidos dos fluxos e comp de onda, deixando a primeira ocorrencia
    waves_norep = np.delete(waves, diff_zero)
    fluxes_norep = np.delete(fluxes, diff_zero)

    return waves_norep, fluxes_norep
    

def get_waves_fluxes(results_filename_hdf5, waves_star_filename, waves_tellurics_filename):

    """
    Dado o nome do arquivo de resultados obtido após correções telúricas,
    retorna comprimento de onda e fluxos da estrela e teluricos

    - results_filename_hdf5: arquivo gerado pela funcao "get_real_spectra", com
    os resultados dos espectros apos correcao telurica
    - waves_star_filename, waves_tellurics_filename: nome do arquivo com os comprimentos de onda 
    da estrela e teluricos
    - wavelengths_star, wavelengths_tellurics: comp de onda do espectro corrigido da estrela e telúrico
    - fluxes_star, fluxes_tellurics: fluxos do espectro corrigido da estrela e telúrico
    """

    print("\n*** Obtaining wavelengths and fluxes ***\n")

    # comprimentos de onda e fluxos considerando valores sobrepostos nas ordens
    wavelengths_star, wavelengths_tellurics = np.array([]), np.array([])
    fluxes_star, fluxes_tellurics = np.array([]), np.array([])

    with h5py.File(results_filename_hdf5, 'r') as hdf: # abrindo
    
        R = np.array(hdf.get('R')) # numero de ordens

        # para todas as ordens, vai colocar nas arrays os comprimentos de onda e fluxos
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
    np.savetxt(waves_star_filename, wavelengths_star.T, delimiter='\n', fmt='%4f')
    np.savetxt(waves_tellurics_filename, wavelengths_tellurics.T, delimiter='\n', fmt='%4f')

    # obtendo comp de onda e fluxos da estrela e telurico, sem as sobreposicoes:
        
    # the run with python takes too long. Running with C is recommended
    print("\n*** Removing superpositions star ***")
    wavelengths_star, fluxes_star = C_correct_superposition_orders(wavelengths_star, fluxes_star, waves_star_filename)
    #wavelengths_tellurics, fluxes_tellurics = python_correct_superposition_orders(wavelengths_tellurics, fluxes_tellurics)
    print("\n*** Removing superpositions tellurics ***")
    wavelengths_tellurics, fluxes_tellurics = C_correct_superposition_orders(wavelengths_tellurics, fluxes_tellurics, waves_tellurics_filename)
    #wavelengths_star, fluxes_star = python_correct_superposition_orders(wavelengths_star, fluxes_star)
    
    print("\n*** Waves and fluxes: OK ***\n")

    # retornando as arays de comp de onda e fluxos da estrela e telurico
    return wavelengths_star, fluxes_star, wavelengths_tellurics, fluxes_tellurics

def find_index(array, value):
    for i in range(len(array)):
        if (array[i] > value-0.1) and (array[i] < value+0.1):
            return i

def run(star_name, data_dir, results_dir):
    """
    Função final para rodar as funções anteriores.

    - star_name: nome da estrela
    - data_dir: caminho para pasta dos dados
    - results_dir: caminho para pasta dos resultados
    """

    print("\n        --------------------------------\n        *** INICIANDO ESTRELA {} ***\n         --------------------------------\n".format(star_name))

    if not os.path.exists(results_dir):  # cria diretorio se nao existe
        os.makedirs(results_dir)

    # nomes dos arquivos para salvar espectros pre-processados(.hdf5), resultados(.hdf5) e rvs(.txt)
    processed_data_filename = results_dir + "/processed_data_" + star_name + ".hdf5"
    results_filename = results_dir + "/results_" + star_name + ".hdf5"
    waves_star_filename = results_dir + "/waves_star_with_reps" + ".csv"
    waves_tellurics_filename = results_dir + "/waves_tellurics_with_reps" + ".csv"
    rvs_filename = results_dir + "/rvs_" + star_name + ".txt"

    # nomes dos arquivos para salvar os espectros da estrela e telurico
    final_spectrum_star = results_dir + "/spectrum_" + star_name
    final_spectrum_tellurics = results_dir + star_name + "_tellurics"

    # pre-processamento dos espectros
    pre_process_data(star_name, data_dir, processed_data_filename)

    # fazendo as correcoes teluricas
    get_real_spectra(processed_data_filename, results_filename, rvs_filename)

    # obtendo comprimento de onda e fluxos dos espectros da estrela e telurico
    waves_star, fluxes_star, waves_tellurics, fluxes_tellurics = get_waves_fluxes(results_filename, waves_star_filename, waves_tellurics_filename)

    # obtendo velocidade radial media
    median_rv, rv_err = get_median_rv(rvs_filename)
    print("Median RV: {}\n".format(median_rv))

    # corrigindo velocidade radial
    c = 299792458
    shift = waves_star * (median_rv/c)
    waves_star = waves_star + shift

    # obtendo SNR medio do espectro da estrela entre os comp de onda ~ 5500 e 5501 A
    idx_ini, idx_end = find_index(waves_star, 5500), find_index(waves_star, 5501)
    SNR = get_SNR(fluxes_star[idx_ini:idx_end]) # usa região em torno de 550 nm (igual HARPS)
    print("Mean SNR: {:.2f}\n".format(SNR))

    # matriz com o espectro
    spectrum = np.array([waves_star, fluxes_star]).T

    # salvando espectros no formato .csv 
    save_spec_csv(final_spectrum_star + ".csv", spectrum)

    # salvando espectro no formato .fits
    save_spec_fits(final_spectrum_star + ".fits", spectrum, SNR, median_rv, rv_err, star_name)

    # plotando espectro
    plot_spectrum(waves_star, fluxes_star, results_dir + "/" + star_name+".pdf", star_name, False) # estrela
    plot_spectrum(waves_tellurics, fluxes_tellurics, results_dir + "/" + star_name + "_tellurics.pdf", star_name, True) # telurico

def main():
    """
    Alterar apenas essa função.
    Mudar nome da estrela ("star_name") e a pasta "data_dir" onde estão os dados e a pasta "results_dir"
    para salvar os resultados.
    """

    star_name = "HD198075"

    # diretorio onde estao os espectros da estrela e ccfs
    data_dir = "/home/giumartos/Servidor/Files/Documents/Mestrado/Trabalho/wobble/Testes/" + star_name

    # diretorio para salvar os dados pre-processados e os resultados
    results_dir = "/home/giumartos/Servidor/Files/Documents/Mestrado/Trabalho/wobble/Teste_HD198075"
    
    run(star_name, data_dir, results_dir)


if __name__ == '__main__':

    # caminho da shared library feita com o C
    so_file = "/home/giumartos/Servidor/Files/Documents/Mestrado/Trabalho/wobble/Testes/where_array.so"
    c_lib = c_init(so_file)  # initialize the C library

    main()
 
