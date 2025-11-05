# model.py

import joblib
import pandas as pd
import numpy as np

# RUTA CORRECTA: Coincide con el nombre del archivo en tu carpeta
MODEL_PATH = "ClasificacionAlphaBetha.pkl" 

class ClassifierModel:
    """
    Clase para cargar y gestionar el modelo de clasificación.
    """
    def __init__(self):
        try:
            self.bundle = joblib.load(MODEL_PATH)
            self.model = self.bundle['model']
            self.scaler = self.bundle['scaler_obj']
            self.best_thr_gb = self.bundle['best_threshold'] 
            self.expected_features = self.bundle['feature_names']
            self.numeric_features = self.bundle['numeric_features']
            print("✅ Modelo y artefactos cargados correctamente.")
        except FileNotFoundError:
            raise Exception(f"Error: No se encontró el archivo del modelo en {MODEL_PATH}")
        except KeyError as e:
            raise Exception(f"Error: La clave {e} no se encontró en el archivo PKL. Verifique el diccionario guardado.")


    def preprocess_and_predict(self, data_json):
        """
        Recibe un JSON, aplica preprocesamiento y realiza la predicción.
        """
        
        df_new = pd.DataFrame([data_json])
        df_new[self.numeric_features] = df_new[self.numeric_features].apply(
            pd.to_numeric, errors='coerce'
        )

        # Si algún valor es nulo, el registro se descarta (limpieza estricta)
        if df_new.isnull().values.any():
            return None 
        
        X_new_processed = pd.DataFrame(0, index=df_new.index, columns=self.expected_features)
        X_new_processed[self.numeric_features] = df_new[self.numeric_features]

        # Mapeo Manual (One-Hot Encoding Replicado)
        X_new_processed['SeniorCity_1'] = (df_new['SeniorCity'] == 1).astype(int)
        X_new_processed['Partner_Yes'] = (df_new['Partner'] == 'Yes').astype(int)
        X_new_processed['Dependents_Yes'] = (df_new['Dependents'] == 'Yes').astype(int)
        X_new_processed['Service1_Yes'] = (df_new['Service1'] == 'Yes').astype(int)
        X_new_processed['Service2_Yes'] = (df_new['Service2'] == 'Yes').astype(int)
        X_new_processed['PaperlessBilling_Yes'] = (df_new['PaperlessBilling'] == 'Yes').astype(int)

        X_new_processed['Security_Yes'] = (df_new['Security'] == 'Yes').astype(int)
        X_new_processed['OnlineBackup_Yes'] = (df_new['OnlineBackup'] == 'Yes').astype(int)
        X_new_processed['DeviceProtection_Yes'] = (df_new['DeviceProtection'] == 'Yes').astype(int)
        
        X_new_processed['TechSupport_Yes'] = (df_new['TechSupport'] == 'Yes').astype(int)
        X_new_processed['TechSupport_No internet service'] = (df_new['TechSupport'] == 'No internet service').astype(int)

        X_new_processed['Contract_One year'] = (df_new['Contract'] == 'One year').astype(int)
        X_new_processed['Contract_Two year'] = (df_new['Contract'] == 'Two year').astype(int)

        X_new_processed['PaymentMethod_Credit card (automatic)'] = (df_new['PaymentMethod'] == 'Credit card (automatic)').astype(int)
        X_new_processed['PaymentMethod_Electronic check'] = (df_new['PaymentMethod'] == 'Electronic check').astype(int)
        X_new_processed['PaymentMethod_Mailed check'] = (df_new['PaymentMethod'] == 'Mailed check').astype(int)
        
        # Escalado
        X_new_processed[self.numeric_features] = self.scaler.transform(X_new_processed[self.numeric_features])

        # Predicción
        y_proba = self.model.predict_proba(X_new_processed)[:, 1][0] 
        y_pred_class = (y_proba >= self.best_thr_gb).astype(int)
        
        class_name = 'Alpha' if y_pred_class == 1 else 'Betha'

        return {
            "Probabilidad_Alpha": round(y_proba, 4),
            "Class": class_name,
            "Umbral_Usado": round(self.best_thr_gb, 4)
        }
