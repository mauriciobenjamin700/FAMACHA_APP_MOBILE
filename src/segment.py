from ultralytics import YOLO
from os.path import (
    basename,
    join
)
from glob import glob
from cv2 import (
    bitwise_and,
    imread, 
    INTER_AREA,
    fillPoly,
    resize,
)
import numpy as np

class Segmentacao:
    
    def __init__(self, path_model_seg='model_segment/weights/best.pt') -> None:
        self.seg_model = YOLO(path_model_seg)
        
    def predict_dir_image(self, list_fname,conf=0.5)->dict:
        """
        Processa uma imagem e retorna um dicionário de dicionários com os dados obtidos.
        Cada chave do dicionário é o nome de uma imagem, dicionário possui as seguintes chaves -> xyxys,confidences,class_id,masks,probs
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
            
        Retorno:
            dic::dict: Dicionário Contendos os dados obtidos no processamento
        """
        
        json = {}
        
        try:
            results = self.seg_model.predict(list_fname,conf=conf,boxes=False,max_det=2)
             
            for idx,result in enumerate(results):
                
                dic = dict()
                boxes = result.boxes.cpu().numpy()

                dic['masks'] = result.masks
                dic['probs'] = result.probs
                dic['xyxys'] = boxes.xyxy
                dic['confidences'] = boxes.conf
                dic['class_id'] = boxes.cls
                
                json[basename(list_fname[idx])] = dic
        except:
            json = None
               
        
        return json
            

    def predict_image(self, image:str,conf:float=0.5):
        """
        Processa uma imagem e retorna um dicionário com os dados obtidos.
        O dicionário possui as seguintes chaves -> xyxys,confidences,class_id,masks,probs
        
        Parâmetros:
            image::str: Uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
            
        Retorno:
            dic::dict: Dicionário Contendos os dados obtidos no processamento
        """
        
        dic = dict()
        
        try:
            results = self.seg_model.predict(image,conf=conf,boxes=False,max_det=2)

            result = results[0]
            boxes = result.boxes.cpu().numpy()
        
            dic['xyxys'] = boxes.xyxy
            dic['confidences'] = boxes.conf
            dic['class_id'] = boxes.cls
            #dic['masks'] = (result.masks.xy,result.masks.data)
            dic['masks'] = result.masks.xy
            
        except:
            dic = None
        
        return dic
            
            
    
    def axis_image(self,fname,confiance=0.5)->list:
        """
        Processa uma imagem e retorna os eixos x1,y1,x2,y2 que compõe os boxs que contem a zona de interesse da imagem.
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            confiance::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
        
        Retorno:
            xyxys::list: Lista contendo tuplas com os eixos da imagem que estão nossa zona de interesse ou
            lista vazia caso não encontre nada
    
        """
        xyxys = []
        try:
            result = self.seg_model.predict(fname,conf=confiance,boxes=False,max_det=2,show_conf=False,show_labels=False)
            
            boxes = result[0].boxes.cpu().numpy()
            
            for xyxy in boxes.xyxy:
            
                xyxys.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])))
        
        except:
            xyxys = None    
        return xyxys
    
    
    
    
    def read_resize(self,fname:str,width=640,height=640)->np.array:
        """

        Args:
            fname (str): _description_
            width (int, optional): _description_. Defaults to 640.
            height (int, optional): _description_. Defaults to 640.

        Returns:
            np.array: Imagem carregada e redimensionada
        """
        img = resize(imread(fname),(width,height),interpolation=INTER_AREA)
        return img
    
    
    def segment_img(self,fname:str):
        """
        Carrega uma imagem da memoria atráves de seu nome de arquivo, segmenta e retorna a zona de interesse coletada após a segmentação.
        
        Parâmetros:
            fname::str: Nome do arquivo que será segmentado
            
        Retorno:
            segmentacao:: numpy array contendo a imagem segmentada a ser retornada ou Nada caso não haja oq segementar na imagem
        """
        
        segmentacao = None
        
        try:
            dados = self.predict_image(fname)
         
            xy = dados["masks"]

            img = self.read_resize(fname)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # Converter a lista de tuplas em um array numpy
            pts = np.array([tuple(map(int, ponto)) for array in xy for ponto in array], dtype=np.int32)

            # Desenhar a região de interesse na máscara
            fillPoly(mask, [pts], (255))  # Preenche a região da máscara com branco

            # Aplicar a máscara na imagem original
            segmentacao = bitwise_and(img, img, mask=mask)
        
        except:
            pass
        
        return segmentacao
    
    def segment_dir_image(self, dir_images:str) -> list:
        """
        Processa um diretório de imagens e retorna uma lista de matrizes, onde cada matriz é uma imagem valida e segmentada
        As imagens invalidas são descartadas durante a segmentação

        Args:
            dir_images::str: Diretório contendo as imagens

        Returns:
            imagens::list: Lista contendo as imagens após a segmentação
        """
        extensoes = ['*.jpg', '*.jpeg', '*.png']
        fnames = []
        

        for i in extensoes:
            fnames += glob(join(dir_images, i))
            
        print(fnames)

        rotulos = []
        imagens = []

        for file in fnames:
            image = self.segment_img(file)

            # Corrigindo a condição para verificar se 'image' não é None
            if image is not None:
                imagens += [image]
                rotulos += [basename(file)]

        return imagens,rotulos
            

if __name__ == "__main__":
    from pathlib import Path
    s = Segmentacao(Path("src/models/YOLO.pt").resolve())
    #cabra = s.read_resize("Imagens/cabra_1.jpg")
    cabra = s.segment_dir_image("Imagens")
    print(cabra)
    #print(s.predict_image(cabra))