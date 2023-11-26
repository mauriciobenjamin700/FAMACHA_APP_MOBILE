import cv2
from ultralytics import YOLO
import numpy as np

class Famacha:

    def __init__(self, path_model='model_segment/weights/best.pt') -> None:
        self.model = YOLO(path_model)
        self.fname = ''
        
    def set_fname(self, fname:str)->None:
        self.fname = fname
    
    def get_fname(self):
        return self.fname
            

    def predict_image(self,conf:float=0.5)-> dict:
        """
        Processa uma imagem e retorna um dicionário com os dados obtidos.
        O dicionário possui as seguintes chaves -> xyxys,confidences,class_id,masks,probs
        
        Parâmetros:
            conf::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
            
        Retorno:
            dic::dict: Dicionário Contendos os dados obtidos no processamento
        """
        
        dic = dict()
        
        try:
            results = self.model.predict(self.fname,conf=conf,boxes=False,max_det=2)

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

        
    def axis_image(self,fname,conf=0.5)->list:
        """
        Processa uma imagem e retorna os eixos x1,y1,x2,y2 que compõe os boxs que contem a zona de interesse da imagem.
        
        Parâmetros:
            fname::str: Nome de uma imagem processada para o recorte
            conf::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
        
        Retorno:
            xyxys::list: Lista contendo tuplas com os eixos da imagem que estão nossa zona de interesse ou
            lista vazia caso não encontre nada
    
        """
        xyxys = []
        try:
            result = self.model.predict(fname,conf=conf,boxes=False,max_det=2,show_conf=False,show_labels=False)
            
            boxes = result[0].boxes.cpu().numpy()
            
            for xyxy in boxes.xyxy:
            
                xyxys.append((int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])))
        
        except:
            xyxys = None    
        return xyxys
    
    #recorta a imagem
    def snip_img(self, conf:float=0.5)->np.ndarray:
        """
        Processa uma imagem e retorna os pixels recortados da imagem.
        Onde os pixels compõe zonas de interesse que a imagem pode vir a possuir
        
        Parâmetros:
            
            conf::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            o valor de confiança pode varia entre 0 e 1.
        
        Retorno:
            interest_region::list: Lista contendo as partes da imagem que estão nossa zona de interesse ou
            lista vazia caso não encontre nada
    
        """
        interest_region = []
        try:
            
            image = cv2.imread(self.fname)
            
            xyxys = self.axis_image(fname=self.fname,confiance=conf)
            if len(xyxys) > 0:
                for xyxy in xyxys:
                    x1,y1,x2,y2 = xyxy
                    interest_region.append(image[y1:y2, x1:x2])
        except:
            interest_region = None
            
        return interest_region
    
    
    def resize(self,width=640,height=640)->np.ndarray:
        img = cv2.resize(cv2.imread(self.fname),(width,height),interpolation=cv2.INTER_AREA)
        return img
    
    
    def segment_img(self,conf:float=0.5)->np.ndarray:
        """
        Recebe uma imagem famacha, a segmenta e retorna a zona de interesse coletada após a segmentação.
        
        Parâmetros:
            conf::float: Grau de confiança que a rede usará para decidir as zonas de recorte,
            
        Retorno:
            segmentacao:: numpy array contendo a imagem segmentada a ser retornada ou Nada caso não haja oq segementar na imagem
        """
        
        segmentacao = None
        
        try:
            dados = self.predict_image(conf)
         
            xy = dados["masks"]

            img = cv2.imread(self.fname)

            mask = np.zeros(img.shape[:2], dtype=np.uint8)

            # Converter a lista de tuplas em um array numpy
            pts = np.array([tuple(map(int, ponto)) for array in xy for ponto in array], dtype=np.int32)

            # Desenhar a região de interesse na máscara
            cv2.fillPoly(mask, [pts], (255))  # Preenche a região da máscara com branco

            # Aplicar a máscara na imagem original
            segmentacao = cv2.bitwise_and(img, img, mask=mask)
        
        except:
            pass
        
        return segmentacao
    
                