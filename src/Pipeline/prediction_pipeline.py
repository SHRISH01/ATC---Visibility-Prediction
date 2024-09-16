import sys 
import pandas as pd
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)
from src.exception  import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,feature):
        try:
            model_path =  "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled =preprocessor.transform(feature)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys) from e

class CustomData:
    def __init__(self,
                DRYBULBTEMPF : float,
				RelativeHumidity : float,
				WindSpeed : float,
				WindDirection : float,
				StationPressure : float, 
				Precip : float):
        self.DRYBULBTEMPF = DRYBULBTEMPF
        self.RelativeHumidity = RelativeHumidity
        self.WindSpeed = WindSpeed
        self.WindDirection = WindDirection
        self.StationPressure = StationPressure
        self.Precip = Precip

    def  get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "DRYBULBTEMPF":[self.DRYBULBTEMPF],
                "RelativeHumidity":[self.RelativeHumidity],
                "WindSpeed":[self.WindSpeed],
                "WindDirection":[self.WindDirection],
                "StationPressure":[self.StationPressure],
                "Precip":[self.Precip]
            }
            return pd.DataFrame(custom_data_input_dict)
        except  Exception as e:
            raise CustomException(e,sys)
        

        
        
    