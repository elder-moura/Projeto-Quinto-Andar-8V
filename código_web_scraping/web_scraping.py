# =====================================================================
# SCRIPT DE EXTRAÇÃO DE DADOS (WEB SCRAPING) - VERSÃO PARA REPOSITÓRIO
# =====================================================================

# Célula 1: Importações
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from bs4 import BeautifulSoup
import re
import time
import pandas as pd
import numpy as np
import io

# Célula 2: Configuração do webdriver
chrome_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=chrome_service)

# Célula 3: URL (Ocultada para fins de privacidade/estritamente acadêmicos)
url = 'URL_ALVO_OCULTA'

# Célula 4: Carregamento da página inicial de listagem
driver.get(url)
print("Página de listagem carregando... Aguardando 5 segundos.")
time.sleep(5)


# Célula 5: Fase 1 - ATUALIZA a lista de URLs (com Checkpoint de salvamento)

# --- CONFIGURAÇÃO --
target_novas_urls = 800  # Defina quantas NOVAS URLs você quer encontrar
segundos_espera_clique = 4 
master_url_file = 'master_url_list.txt'
url = 'URL_ALVO_OCULTA'
# --------------------

# --- 1. CARREGA A LISTA MESTRE ANTIGA ---
urls_ja_coletadas = set()
try:
    with open(master_url_file, 'r') as f:
        urls_ja_coletadas = set([line.strip() for line in f.readlines() if line.strip()])
    print(f"Lista mestre existente carregada. {len(urls_ja_coletadas)} URLs já conhecidas.")
except FileNotFoundError:
    print("Nenhuma lista mestre encontrada. Começando uma nova.")

# --- 2. ABRE O NAVEGADOR ---
print("Abrindo navegador para coletar novas URLs...")
chrome_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=chrome_service)
driver.get(url)
time.sleep(5) 

# --- 3. LOOP PARA COLETAR NOVAS URLs ---
novas_urls_nesta_sessao = 0
cliques_desde_ultimo_save = 0
total_cliques = 0

print(f"Iniciando coleta de {target_novas_urls} NOVAS URLs...")

# Lidar com pop-up de cookie (se houver)
try:
    cookie_button = driver.find_element(By.ID, "cookie-notifier-cta")
    if cookie_button:
        driver.execute_script("arguments[0].click();", cookie_button)
        print("Pop-up de cookie aceito.")
        time.sleep(2)
except:
    print("Nenhum pop-up de cookie encontrado.")

while novas_urls_nesta_sessao < target_novas_urls:
    try:
        ver_mais_button = driver.find_element(By.ID, "see-more")
        
        # Rola o botão para a PARTE DE BAIXO da tela
        driver.execute_script("arguments[0].scrollIntoView(false);", ver_mais_button)
        time.sleep(1) 
        
        # Usa clique via JavaScript (mais robusto contra sobreposição de elementos)
        driver.execute_script("arguments[0].click();", ver_mais_button)
        
        total_cliques += 1
        cliques_desde_ultimo_save += 1
        time.sleep(segundos_espera_clique)
        
        temp_soup = BeautifulSoup(driver.page_source, 'html.parser')
        endereco_tags = temp_soup.find_all('h2', class_='_72Hu5c')
        
        novas_encontradas_agora = 0
        for tag in endereco_tags:
            link_tag = tag.find_parent('a')
            if link_tag and 'href' in link_tag.attrs:
                url_parcial = link_tag['href']
                if url_parcial.startswith('/imovel'):
                    url_completa = "https://www.SITE_OCULTO.com.br" + url_parcial
                    
                    if url_completa not in urls_ja_coletadas:
                        urls_ja_coletadas.add(url_completa) 
                        novas_urls_nesta_sessao += 1
                        novas_encontradas_agora += 1
        
        print(f"   ... (Clique {total_cliques}) Encontrou {novas_encontradas_agora} novas. Total da sessão: {novas_urls_nesta_sessao} / {target_novas_urls} ...")
            
        # --- CHECKPOINT DE SALVAMENTO (A CADA 10 CLIQUES) ---
        if cliques_desde_ultimo_save >= 10:
            print(f"--- CHECKPOINT: Salvando {len(urls_ja_coletadas)} URLs totais no arquivo... ---")
            try:
                with open(master_url_file, 'w') as f:
                    for url_item in urls_ja_coletadas:
                        f.write(f"{url_item}\n")
                cliques_desde_ultimo_save = 0 
            except Exception as e:
                print(f"   !!! ERRO AO SALVAR CHECKPOINT: {e} !!!")
            
    except NoSuchElementException:
        print("Botão 'Ver mais' não encontrado. Chegamos ao fim da página.")
        break
    except Exception as e:
        print(f"Ocorreu um erro ao tentar clicar ou extrair links: {e}")
        print("--- ERRO FATAL: O navegador pode ter travado. ---")
        break 

# --- 4. SALVAMENTO FINAL FASE 1 ---
print(f"\nFase 1 Concluída (ou interrompida).")
print(f"Adicionadas {novas_urls_nesta_sessao} novas URLs.")
print(f"Salvando lista mestre final com {len(urls_ja_coletadas)} URLs totais em '{master_url_file}'")
try:
    with open(master_url_file, 'w') as f:
        for url_item in urls_ja_coletadas:
            f.write(f"{url_item}\n")
except Exception as e:
    print(f"\nERRO ao salvar lista mestre final: {e}")

driver.quit()
print("Driver fechado. Fase 1 (Coleta de URLs) finalizada.")


# Célula 6: Fase 2 - Processa URLs em Lotes (com extração detalhada e Checkpoint)

# --- CONFIGURAÇÃO DOS LOTES ---
LIMITE_POR_SESSAO = 200  
SEGUNDOS_ESPERA_PAGINA = 5 
# --- ARQUIVOS DE CONTROLE ---
master_url_file = 'master_url_list.txt'
backup_csv_file = 'backup_imoveis_processados.csv'

# --- 1. RE-ABRE O NAVEGADOR ---
print("Iniciando novo lote... Abrindo o navegador.")
chrome_service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=chrome_service)

# --- 2. CARREGA A LISTA MESTRE ---
try:
    with open(master_url_file, 'r') as f:
        all_urls = [line.strip() for line in f.readlines()]
    print(f"Lista mestre com {len(all_urls)} URLs carregada.")
except FileNotFoundError:
    print(f"Erro: Arquivo '{master_url_file}' não encontrado. Rode a Célula 5 primeiro.")
    driver.quit()
    raise

# --- 3. CARREGA O BACKUP (CHECKPOINT DE RESILIÊNCIA) ---
lista_de_dados = []
urls_ja_feitas = set()
try:
    df_backup = pd.read_csv(backup_csv_file)
    lista_de_dados = df_backup.to_dict('records') 
    urls_ja_feitas = set(df_backup['URL']) 
    print(f"Backup encontrado. {len(urls_ja_feitas)} URLs já foram processadas.")
except FileNotFoundError:
    print("Arquivo de backup não encontrado. Começando do zero.")
except pd.errors.EmptyDataError:
    print("Arquivo de backup encontrado, mas está vazio. Começando do zero.")

# --- 4. CRIA A LISTA "A FAZER" PARA ESTE LOTE ---
urls_para_visitar_nesta_sessao = [url for url in all_urls if url not in urls_ja_feitas]\n
total_restante = len(urls_para_visitar_nesta_sessao)

if total_restante == 0:
    print("Todos os imóveis da lista mestre já foram processados.")
    driver.quit()
else:
    urls_para_visitar_nesta_sessao = urls_para_visitar_nesta_sessao[:LIMITE_POR_SESSAO]
    total_lote = len(urls_para_visitar_nesta_sessao)
    print(f"Processando um lote de {total_lote} imóveis (de {total_restante} restantes)...")

    # --- 5. LOOP PRINCIPAL DE DETALHAMENTO DE CONTEÚDO ---
    for i, url_imovel in enumerate(urls_para_visitar_nesta_sessao):
        
        print(f"Processando {i+1}/{total_lote} (Total geral: {len(lista_de_dados)+1}/{len(all_urls)}): [MASCARADO]")
        
        try:
            driver.get(url_imovel)
            time.sleep(SEGUNDOS_ESPERA_PAGINA) 
            soup_imovel = BeautifulSoup(driver.page_source, 'html.parser')
            imovel_data = {'URL': url_imovel}

            # Bloco 1: Endereço (com higienização e separação estruturada)
            imovel_data['Rua'] = np.nan
            imovel_data['Bairro'] = np.nan
            imovel_data['Cidade'] = np.nan
            try:
                map_wrapper = soup_imovel.find('div', {'data-testid': 'smallMapWrapper'})
                if map_wrapper:
                    rua_tag = map_wrapper.find('h4', class_='EqjlRj')
                    if rua_tag: imovel_data['Rua'] = rua_tag.text.strip()
                    bairro_tag = map_wrapper.find('small', class_='pwAPLE')
                    if bairro_tag:
                        partes = bairro_tag.text.strip().split(',')
                        imovel_data['Bairro'] = partes[0].strip()
                        if len(partes) > 1: imovel_data['Cidade'] = partes[1].strip()
            except: pass 

            # Bloco 2: Mapeamento de Custos e Atributos Financeiros
            imovel_data['Aluguel'] = np.nan
            imovel_data['Condominio'] = np.nan
            imovel_data['IPTU'] = np.nan
            imovel_data['Total'] = np.nan
            try:
                price_table = soup_imovel.find('ul', {'data-testid': 'listing-price-table'})
                if price_table:
                    for item in price_table.find_all('li'):
                        text_parts = [t.text.strip() for t in item.find_all(['p', 'h4'])]
                        if len(text_parts) >= 2:
                            label, value = text_parts[0], text_parts[-1]
                            if 'Aluguel' in label: imovel_data['Aluguel'] = value
                            elif 'Condomínio' in label: imovel_data['Condominio'] = value
                            elif 'IPTU' in label: imovel_data['IPTU'] = value
                            elif 'Total' in label: imovel_data['Total'] = value
            except: pass

            # Bloco 3: Especificações Estruturais do Imóvel
            imovel_data['Metragem'] = np.nan
            imovel_data['Quartos'] = np.nan
            imovel_data['Banheiros'] = np.nan
            imovel_data['Vagas'] = np.nan
            imovel_data['Andar'] = np.nan
            imovel_data['Pet'] = np.nan
            imovel_data['Mobilia'] = np.nan
            try:
                main_info = soup_imovel.find('div', {'data-testid': 'house-main-info'})
                if main_info:
                    for spec in main_info.find_all('div', class_='MainInfo_iconDescriptionWrapper__St8RA'):
                        texto_spec = spec.find('p').text.strip()
                        if 'm²' in texto_spec: imovel_data['Metragem'] = texto_spec
                        elif 'quarto' in texto_spec: imovel_data['Quartos'] = texto_spec
                        elif 'banheiro' in texto_spec: imovel_data['Banheiros'] = texto_spec
                        elif 'vaga' in texto_spec: imovel_data['Vagas'] = texto_spec
                        elif 'andar' in texto_spec: imovel_data['Andar'] = texto_spec
                        elif 'pet' in texto_spec: imovel_data['Pet'] = texto_spec
                        elif 'mobília' in texto_spec: imovel_data['Mobilia'] = texto_spec
            except: pass
            
            # Bloco 4: Comodidades Internas
            imovel_data['Comodidades'] = np.nan
            try:
                amenities_div = soup_imovel.find('div', {'data-testid': 'amenities'})
                tags = amenities_div.find_all(['p', 'span'])
                imovel_data['Comodidades'] = ", ".join([t.text.strip() for t in tags if t.text.strip()])
            except: pass

            lista_de_dados.append(imovel_data)

        except Exception as e:
            print(f"   !!! ERRO GERAL ao processar o registro: {e} !!!")
            lista_de_dados.append({'URL': url_imovel, 'Aluguel': f"ERRO: {e}"}) 
        
        # --- 6. SALVA O BACKUP A CADA 25 IMÓVEIS ---
        if (i+1) % 25 == 0 or (i+1) == total_lote:
            print(f"--- Checkpoint: Salvando backup com {len(lista_de_dados)} imóveis TOTAIS ---")
            pd.DataFrame(lista_de_dados).to_csv(backup_csv_file, index=False, encoding='utf-8-sig')

    print(f"\nLote de {total_lote} imóveis concluído.")
    driver.quit()
    print("Driver fechado. Ajuste a janela de execução antes de rodar o próximo lote.")


# Célula 6.5: Célula de Consulta, Auditoria e Verificação
print("--- Iniciando Consulta de Dados do Backup ---")
backup_csv_file = 'backup_imoveis_processados.csv'

try:
    df_consulta = pd.read_csv(backup_csv_file)
    print(f"\nArquivo '{backup_csv_file}' carregado com sucesso.")
    print(f"Total de imóveis no backup: {df_consulta.shape[0]}")
    
    print("\n--- Informações das Colunas (df.info()) ---")
    buffer = io.StringIO()
    df_consulta.info(buf=buffer)
    print(buffer.getvalue())
    
    print("\n--- Últimos 5 imóveis coletados (df.tail()) ---")
    print(df_consulta.tail())

except FileNotFoundError:
    print(f"ERRO: Arquivo de backup '{backup_csv_file}' ainda não foi criado.")
except pd.errors.EmptyDataError:
    print(f"AVISO: O arquivo de backup '{backup_csv_file}' está vazio.")
except Exception as e:
    print(f"Ocorreu um erro ao ler o arquivo: {e}")


# Célula 7: Consolidação do DataFrame final e salvamento em CSV de Produção
print("Criando DataFrame final a partir do arquivo de backup...")

try:
    df_final = pd.read_csv('backup_imoveis_processados.csv')
    
    # Filtragem estrutural: remove registros corrompidos ou com falhas de captura
    df_final = df_final.dropna(subset=['Aluguel'])
    df_final = df_final[~df_final['Aluguel'].astype(str).str.startswith('ERRO')]

    print(f"Total de {len(df_final)} imóveis consolidados com sucesso.")
    print("Amostra dos dados finais:")
    print(df_final.head())

    nome_arquivo = 'dataset_imoveis_consolidado.csv'
    df_final.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')
    print(f"\nDataset final salvo com sucesso em: {nome_arquivo}")

except FileNotFoundError:
    print("Arquivo de backup intermediário não encontrado.")
except Exception as e:
    print(f"Ocorreu um erro na consolidação: {e}")