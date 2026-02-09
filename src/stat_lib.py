#!/usr/bin/python
#!python
# -*- coding: utf-8 -*-
'''
# @Updated on 2026/02/08
# @Created on 2026/02/08
# @author: Flavio Lichtenstein
# @local: CENTD, Instittuto Butantan, Bioinformática & Bioestatística, São Paulo, Brazil
'''

import numpy as np
import os, math
from os.path import join as osjoin
from os.path import exists as exists
import pandas as pd
# from collections import OrderedDict
from typing import List  #  Optional, Iterable, Set, Tuple, Any

# import zipfile, zlib

from statistics import mode
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

def prepare_title(title, maxCount=12):
	if title == None: return None
	title = title.strip()
	if title == "": return "???"

	title = title.replace("/", "-").replace("_", ": ")
	words = title.split(" ")

	stri = ""; count = 0
	for word in words:
		stri += word + " "
		count += 1
		if count == maxCount:
			stri = stri.rstrip() + "\n"
			count = 0

	return stri.rstrip()

def write_txt(text, filename, path="./", to_append=False, verbose=False): # encondig = 'utf-8',
	if not exists(path):
		os.mkdir(path)

	filename = osjoin(path, filename)
	try:
		ret = True
		if to_append:
			h = open(filename, mode = "a+")
		else:
			h = open(filename, mode = "w")
		h.write(text)
	except ValueError:
		print(f"Error '{ValueError}', writing in '{filename}'")
		ret = False
	finally:
		h.close()

	if not ret:
		return False

	if verbose: print(f"File saved: {filename}")
	return True

def read_txt(filename, path="./", iniLine=None, endLine=None, verbose=False):  # encondig = 'utf-8',
	filename = osjoin(path, filename)
	text = []

	try:
		h = open(filename, mode="r")

		if iniLine is not None or endLine is not None:
			iniLine = -1 if iniLine is None else iniLine
			endLine = np.inf if endLine is None else endLine

			count = 0
			while True:
				if count < iniLine:
					_ = h.readline()
				else:
					line = h.readline()
					if not line : break

					text.append(line.replace('\n','') )

				count += 1
				if count > endLine: break
		else:
			text = [line.replace('\n','') for line in h.readlines()]

		h.close()
		if verbose: print("File read at '%s'"%(filename))
	except ValueError:
		print(f"Error '{ValueError}' while reading '{filename}'")

	return "\n".join(text)

# encondig = 'utf-8',
def pdreadcsv(filename:str, path:str="./", sep:str='\t', dtype:dict={}, colnames:List=[], skiprows:int=0,
			  selcols:List=[], sortcols:List=[], low_memory:bool=False, removedup:bool=False, header:int=0,
			  verbose:bool=False) -> pd.DataFrame:

	if not exists(path):
		print(f"Path does not exists: '{path}'")
		return pd.DataFrame()

	filename = osjoin(path, filename)

	if not exists(filename):
		print(f"File does not exist: '{filename}'")
		return pd.DataFrame()

	try:
		if dtype == {}:
			df = pd.read_csv(filename, sep=sep, low_memory=low_memory, skiprows=skiprows, header=header)  # , encondig = encondig
		else:
			df = pd.read_csv(filename, sep=sep, dtype = dtype, low_memory=low_memory, skiprows=skiprows, header=header) #, encondig = encondig

	except:
		print(f"Error reading csv/tsv '{filename}'")
		return pd.DataFrame()

	try:
		if len(colnames) > 0: df.columns = colnames
		if len(selcols)  > 0: df = df[  selcols ]
		if len(sortcols) > 0: df = df.sort_values(sortcols)
		if removedup: df = df.drop_duplicates()
	except ValueError:
		print(f"Error '{ValueError}' columns/selecting/sorting '{filename}'")
		return df

	if verbose: print("Table opened (%s) at '%s'"%(df.shape, filename))
	return df

# encondig = 'utf-8',
def pdwritecsv(df, filename, path="./", sep='\t', index=False, verbose=False):

	if not exists(path):
		print("Path does not exists: '%s'"%(path))
		return False

	filename = osjoin(path, filename)
	try:
		df.to_csv(filename, sep=sep, index=index)  # , encondig = encondig
	except ValueError:
		print(f"Error '{ValueError}', writing in '{filename}'")
		return False

	if verbose: print("Table saved (%s) at '%s'"%(df.shape, filename))
	return True


def month_to_num(stri):
	if isinstance(stri, int): return stri
	if not isinstance(stri, str): return stri

	stri2 = stri.lower()

	if stri2 == 'jan':
		return 1
	if stri2 == 'feb' or stri2 == 'fev':
		return 2
	if stri2 == 'mar':
		return 3
	if stri2 == 'apr' or stri2 == 'abr':
		return 4
	if stri2 == 'may' or stri2 == 'mai':
		return 5
	if stri2 == 'jun':
		return 6
	if stri2 == 'jul':
		return 7
	if stri2 == 'aug' or stri2 == 'ago':
		return 8
	if stri2 == 'sep' or stri2 == 'set':
		return 9
	if stri2 == 'oct' or stri2 == 'out':
		return 10
	if stri2 == 'nov':
		return 11
	if stri2 == 'dec' or stri2 == 'dez':
		return 12

	return stri


def criar_dados_normais(MU, SSD, N):
    samp = np.random.normal(loc=MU, scale=SSD, size=N)
    return samp

# metodo estatístico
def calc_estatistica_descritiva(lista: list, verbose:bool=False):

	N = len(lista)

	mini = np.min(lista)
	maxi = np.max(lista)

	mu = np.mean(lista)
	med = np.median(lista)
	mod = mode(lista)

	ssd = np.std(lista)

	n = len(lista)
	n4 = int(n/4)

	lista = list(lista)
	lista.sort()

	mini = int( lista[0] )
	maxi = int( lista[-1] )

	q1, q2, q3 = np.quantile(lista, [0.25, 0.5, 0.75])

	s_quantile = f"mínimo {mini}, Q1 {q1}, mediana {q2}, Q3 {q3}, máximo {maxi}"

	stri = f"N={N}, média={mu:.2f} mediana={q2:.2f} moda={mod:.2f} e ssd={ssd:.2f}"
	if verbose:
		print(stri)
		print(s_quantile)

	return n, mu, q1, q2, q3, mod, ssd, mini, maxi, stri, s_quantile

def calc_tamanho_efeito(n1:int, mu1:float, ssd1:float, 
                        n2:int, mu2:float, ssd2:float, verbose:bool=False):

	ssd_pool = math.sqrt( ((n1-1)*ssd1**2 + (n2-1)*ssd2**2) / (n1+n2-2) )
	diff = mu2-mu1

	EffSize = diff / ssd_pool
	stri = f"O tamanhao de efeito é de {EffSize:.2f} para uma diferença de {diff:.2f} e o ssd conjunto de {ssd_pool:.2f}"
	if verbose:
		print(stri)

	return EffSize, diff, ssd_pool, stri

def stat_asteristics(pval, NS='NS'):
	if pval >= 0.05:   return NS
	if pval > 0.01:    return '*'
	if pval > 0.001:   return '**'
	if pval > 0.0001:  return '***'
	return '****'

def calc_ttest_independente(samp1:list, samp2:list, equal_var:bool=True, alpha:float=0.05):

	# t-test independente
	t, pval = stats.ttest_ind(samp2, samp1, equal_var=equal_var)

	msg = f"Teste-t independente = estatística {t:.3f} e p-valor {pval:.2e}"

	if pval >= alpha:
		msg += '\n' + "Aceitamos a Hipótese nula, não houve efeicalc_normalidade_SWTto significativo."
	else:
		msg += '\n' + "Rejeitamos a Hipótese nula, houve efeito significativo."

	return t, pval, msg   
    
def calc_normalidade_SWT(sample, alpha = 0.05, NS='NS'):
	# teste de normalidade de Shapiro-Wilkis
	stat, pvalue = stats.shapiro(sample)

	if pvalue > alpha:
		text = 'Segundo o teste de Shapiro-Wilk a distribuição se assemelha a uma distribuição normal (aceita-se H0)'
		ret = True
	else:
		text = 'Segundo o teste de Shapiro-Wilk a distribuição não se assemelha a uma distribuição normal (rejeita-se H0)'
		ret = False

	s_ater = stat_asteristics(pvalue)
	text_stat = f'p-value {pvalue:.2e} ({s_ater})'

	return ret, text, text_stat, stat, pvalue
    
def plot_2_distribuições_normais(MUs, SSDs, N1, N2, colors = ['blue', 'red'], figsize=(9, 6)):

	assert len(MUs)==2, 'Enviar 2 médias'
	assert isinstance(MUs, list), 'Enviar 2 médias na forma lista'

	assert len(SSDs)==2, 'Enviar 2 desvios padrões amostrais'
	assert isinstance(SSDs, list), 'Enviar 2 desvios padrões amostrais na forma lista'

	MU1, SSD1 = MUs[0], SSDs[0]
	MU2, SSD2 = MUs[1], SSDs[1]

	samp1 = criar_dados_normais(MU1, SSD1, N1)
	samp2 = criar_dados_normais(MU2, SSD2, N2)

	n1, mu1, q11, med1, q21, mod1, ssd1, mini1, maxi1, stri1, s_quantile1, = \
	calc_estatistica_descritiva(samp1, verbose=False)

	n2, mu2, q12, med2, q22, mod2, ssd2, mini2, maxi2, stri2, s_quantile2, = \
	calc_estatistica_descritiva(samp2, verbose=False)

	ES, diff_ES, ssd_pool, stri_ES = calc_tamanho_efeito(n1, mu1, ssd1, n2, mu2, ssd2)

	fig, ax = plt.subplots(figsize=figsize)

	title = ''

	for i in range(2):
		color = colors[i]

		label = 'dist%d'%(i+1)

		if i == 0:
			mu, ssd, n = mu1, ssd1, n1
			samples = samp1
		else:
			mu, ssd, n = mu2, ssd2, n2
			samples = samp2

		
		ax = sns.histplot(samples, stat='density', color=color, alpha=0.3, label=label, ax=ax)
		sns.rugplot(samples, color=color, alpha=0.1, ax=ax)

		max_y = max(p.get_height() for p in ax.patches)

		# Criando o eixo x        
		seqx = np.linspace(stats.norm.ppf(0.001, MUs[i], SSDs[i]), stats.norm.ppf(0.999, MUs[i], SSDs[i]), 100)

		# fiting da curva teorica
		normal_pdf = stats.norm.pdf(seqx, MUs[i], SSDs[i])
		sns.lineplot(x=seqx, y=normal_pdf, color=colors[i])

		
		ax.axvline(x=mu, ymin=0, ymax=max_y, color=color)
		ax.axvline(x=mu+ssd, ymin=0, ymax=max_y/2, color=color, linestyle='--')
		ax.axvline(x=mu-ssd, ymin=0, ymax=max_y/2, color=color, linestyle='--')

		ret, text, text_stat, stat, pvalue = calc_normalidade_SWT(samples, alpha = 0.05, NS='NS')

		if title != '':
			title += '\n'

		title += f"A distribuição {i+1} ({color}) tem media(SSD) = {mu:.2f} ({ssd:.2f}) e N={n}"
		title += '\n' + text + ' -> ' + text_stat

	title += f'\nTamanho de efeito: {ES:.2f}, diferença={diff_ES:.2f}, ssd conjunto={ssd_pool:.2f}'

	diff_ssd = np.abs(ssd1 - ssd2)
	equal_var = diff_ssd <= ssd1*.50
	t, pval, ttest_msg  = calc_ttest_independente(samp1, samp2, equal_var=equal_var, alpha=0.05)

	title += f'\nt-test independente: {ttest_msg}'

	ax.set_xlabel("values")
	ax.set_ylabel("percentage (%)")
	ax.set_title(title)
	plt.grid()
	plt.legend();


def join_series(samp1, samp2):
    dic = {'val': samp1, 'grupo': 'samp1'}
    df = pd.DataFrame(dic)

    dic = {'val': samp2, 'grupo': 'samp2'}
    df2 = pd.DataFrame(dic)

    df = pd.concat([df, df2])

    df.reset_index(inplace=True, drop=True)
    
    return df

def join_3series(samp1, samp2, samp3):
	dic = {'val': samp1, 'grupo': 'samp1'}
	df = pd.DataFrame(dic)

	dic = {'val': samp2, 'grupo': 'samp2'}
	df2 = pd.DataFrame(dic)

	dic = {'val': samp3, 'grupo': 'samp3'}
	df3 = pd.DataFrame(dic)

	df = pd.concat([df, df2, df3])

	df.reset_index(inplace=True, drop=True)

	return df



#-- função fi --> calcula o Gamma que um fator de "confianca" (~95%) da estatística
# disribuição bi-caudal --> correto - intervalo de confianca

# confianca é de 95% --> alpha = 0.05
# confianca é de 99% --> alpha = 0.01


def calc_intervalo_confianca(samp1, samp2, confianca=.95):
    
    n1, n2 = len(samp1), len(samp2)
    
    mu1 = np.mean(samp1); ssd1 = np.std(samp1, ddof=1)
    mu2 = np.mean(samp2); ssd2 = np.std(samp2, ddof=1)
    
    _, p_lev = stats.levene(samp1, samp2)
    equal_var = p_lev > 0.05
    
    # Welch-Satterthwaite
    dof = (ssd1**2/n1 + ssd2**2/n2)**2 / ( (ssd1**2/n1)**2/(n1-1) + (ssd2**2/n2)**2/(n2-1) )
    
    diff = mu2 - mu1
    SEM = np.sqrt(ssd1**2/n1 + ssd2**2/n2)
    
    ssd_pool = math.sqrt( ((n1-1)*ssd1**2 + (n2-1)*ssd2**2) / (n1+n2-2) )
    effect_size = diff / ssd_pool
    
    alpha = 1 - confianca
    tcrit = stats.t.ppf(1 - alpha/2, dof)
    CI = (diff - tcrit*SEM, diff + tcrit*SEM)
    
    _, pval_ttest, stri_ttest = calc_ttest_independente(samp1, samp2, equal_var=equal_var)
    
    return CI, SEM, n1, n2, effect_size, diff, pval_ttest, stri_ttest, mu1, mu2, ssd1, ssd2, ssd_pool

def test_one_way_ANOVA5 (samp1, samp2, samp3, samp4, samp5, alpha = 0.05):
	# teste de variancias de Fisher - one way ANOVA (analysis of variance)
	stat, pvalue = stats.f_oneway(samp1, samp2, samp3, samp4, samp5)

	if pvalue > alpha:
		text = 'As distribuições têm médias similares (não se rejeita a H0)'
		ret = True
	else:
		text = 'As distribuições não têm médias similares (rejeita-se a H0)'
		ret = False

	text_stat = 'p-value %.2e (%s)'%(pvalue, stat_asteristics(pvalue))

	return ret, text, text_stat, stat, pvalue