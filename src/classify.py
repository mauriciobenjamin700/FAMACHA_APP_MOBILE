from numpy import (
    mean, 
    median,
    std
)
from pickle import load
from pandas import DataFrame
from os.path import join,expanduser


class Classificacao:
    
    def __init__(self, path_model_class='model_classific/RF_Model.pkl') -> None:
        self.clas_model = self.loadModel(path_model_class)
        

    def loadModel(self,name_model='Modelo.pkl'):
        """
        Carregar o modelo RandomForestClassifer salvo em um arquivo pkl e o retorna.
        
        Args:
            model_name::str: Nome do arquivo que será gerado
            
        Return:
            model::RF: Modelo RandomForestClassifer já treinado.
        """ 
        # 
        with open(name_model, 'rb') as arquivo:
            model = load(arquivo)
            return model
        
    def predict(self, imagem_RGB):
        
        novo_dado = self.extract_one(imagem_RGB)

        previsao = self.clas_model.predict(novo_dado)

        return bool(previsao)


    def extract_one(self,imagem_RGB)->DataFrame:
        """
        Extrai as caracteristica de uma image que teve seu caminho passado como parâmetro e 
        arquiva os resultados em um DF

        Args:
            fname::str: Nome da imagem que será aberta e processada.

        Returns:
            df::DataFrame: DataFrame pandas contendo os resultados da extração.
        """

        lista_dados = []

        colunas = ['Media_Canal_R','Media_Canal_G','Media_Canal_B','Mediana_Canal_R','Mediana_Canal_G','Mediana_Canal_B','Desvio_Canal_R','Desvio_Canal_G','Desvio_Canal_B']

        media_R = round(mean(imagem_RGB[:,:,0]))
        media_G = round(mean(imagem_RGB[:,:,1]))
        media_B = round(mean(imagem_RGB[:,:,2]))

        mediana_R = round(median(imagem_RGB[:,:,0]))
        mediana_G = round(median(imagem_RGB[:,:,1]))
        mediana_B = round(median(imagem_RGB[:,:,2]))

        desvio_R = round(std(imagem_RGB[:,:,0]))
        desvio_G = round(std(imagem_RGB[:,:,1]))
        desvio_B = round(std(imagem_RGB[:,:,2]))


        lista_dados.append([media_R,media_G,media_B,mediana_R,mediana_G,mediana_B,desvio_R,desvio_G,desvio_B])

        df = DataFrame(data=lista_dados,columns=colunas)

        return df
    
    def extract_all(self, images:list, name_images:list)->DataFrame:
        """
        Extrai as caracteristica de uma base de dados passada como parâmetro e 
        arquiva os resultados em um DF
        
        Args:
            images::list: Lista contendo as imagens já segmentadas para a extração de caracteristicas
            name_images::list: Lista contendo o nome de cada image
        
        Return:
            df::DataFrame: DataFrame pandas contendo os resultados da extração.
        """

        lista_dados = []

        colunas = ["IMAGEM", "SITUAÇÃO"]
        
        lista_dados = []

        #não está segmentando, apenas classificando, seria adequado segmentar no processo
        for id,image in enumerate(images):


            if (self.predict(image)) == True:
                lista_dados.append([name_images[id],"SAUDÁVEL"])
            else:
                lista_dados.append([name_images[id],"DOENTE"])
                
        df = DataFrame(data=lista_dados,columns=colunas)

        return df
    
    def export(self,pasta:list, rotulos:list,mode:str="excel",output_filename="resultado", output="Documents")->int:
        """
        Processa uma página de imagens FAMACHA e salva o resultado no formato escolhido pelo usuário. 
        Opções válidas para mode [csv, excel, json]

        Args:
            pasta::list: imagens segmentadas da pasta
            rotulos::str: lista contendo os nomes de identificação de cada imagem
            output_filename::str: Nome e caminho de saida para o arquivo que será gerado.
            mode::str: Palavra chave que define como serão salvos os dados processados [csv,excel,json].
            output::str:"pasta que será salvo o arquivo"
    
        Return:
            sinal::int: 0 para falha e 1 para sucesso
        """
        sinal = 0
        to_save = join(expanduser('~'),output)
        
            
        if(len(pasta)) > 0:
            sinal = 1
            df = self.extract_all(pasta,rotulos)
            output_filename = join(to_save,output_filename)
            match mode.lower():
                case 'csv':
                    df.to_csv(output_filename+'.csv',index=False)
                
                case 'excel':
                    df.to_excel(output_filename+'.xlsx',index=False)
                
                case 'json':
                    df.to_json(output_filename+'.json',orient='records')

            
        
        return sinal

if __name__ == "__main__":
    from pathlib import Path
    
    c = Classificacao(Path("src/models/RF.pkl").resolve())
    print(c.export("Imagens","csv"))
    