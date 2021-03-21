import pickle 
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import tensorflow
from PIL import Image
from scipy.io import wavfile
import librosa
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
  

#Welcome the patients
path = '/content/drive/MyDrive/hacknitr/'
while(1):
        print("--"*60,sep='\n')
        print(("Invincible Gut").center(120),sep='\n')
        print("--"*60,sep='\n')
        print(("Diagnostics at your door step").center(120),sep='\n')
        print("--"*60,sep='\n')
        print("\n")
        print("List of predictors",sep='\n')
        print("1. Symptoms predictor")
        print("2. Diabetes",sep='\n')
        print("3. Malaria",sep='\n')
        print("4. Thyroid ",sep='\n')
        print("5. Pneumonia",sep='\n')
        print("6. Insect bite",sep='\n')
        print("7. COVID-19",sep='\n')
        print("\n")
        predictor= int(input("Please choose your desired predictor number: "))
        print("\n")

        if type(predictor)!=int or 1<predictor>8:
            print("Please enter a number between 1-7")
            continue

        if predictor == 1: # Symptoms
            print("**"*60)
            print(("Prediction using symptoms").center(120))
            print("**"*60)
            print("\n")
            print("Enter the indices of the symptoms you have from the followinf list as one line: ")
            symptoms = ['swelled_lymph_nodes',
                'brittle_nails',
                'runny_nose',
                'cough',
                'loss_of_smell',
                'continuous_sneezing',
                'nausea',
                'skin_rash',
                'malaise',
                'abnormal_menstruation',
                'dizziness',
                'redness_of_eyes',
                'muscle_weakness',
                'polyuria',
                'irritability',
                'sinus_pressure',
                'blackheads',
                'congestion',
                'weight_loss',
                'high_fever',
                'weight_gain',
                'restlessness',
                'throat_irritation',
                'obesity',
                'excessive_hunger',
                'depression',
                'pus_filled_pimples',
                'enlarged_thyroid',
                'chills',
                'fast_heart_rate',
                'vomiting',
                'chest_pain',
                'lethargy',
                'sweating',
                'fatigue',
                'muscle_pain',
                'diarrhoea',
                'headache',
                'irregular_sugar_level',
                'phlegm',
                'puffy_face_and_eyes',
                'swollen_extremeties',
                'mood_swings',
                'increased_appetite',
                'scurring',
                'rusty_sputum',
                'blurred_and_distorted_vision',
                'cold_hands_and_feets',
                'breathlessness']
            diseases = ['Acne','Common Cold','Diabetes ','Heart attack','Hyperthyroidism','Hypothyroidism','Malaria','Pneumonia']

            for i in range(1, len(symptoms)+1):
                print(str(i),'.',symptoms[i-1])

            s = [int(x) for x in input("Enter symptoms indices:").split()]
            s_input = [[0]*len(symptoms)]
            for i in s:
                s_input[0][i-1] = 1

            model=tensorflow.keras.models.load_model(path+'disease.h5')

            prediction=model.predict(s_input)

            print("Prediction is completed")
            print("\n")

            p = np.argmax(prediction[0])
            print("You run a high chance of being affect by", diseases[p])
            print("Enter the particulars in the disease-specific page to get a better diagnosis.")
            continue

        if predictor==2: # Diabetes

            print("**"*60)
            print(("Diabetes prediction").center(120))
            print("**"*60)
            print("\n")
            print("Please Enter the following information: ")

            pregnancies= int(input("Enter number of pregnancies: "))
            glucose= int(input("Enter your glucose level (mg/dl): "))
            blood_pressure= int(input("Enter your blood pressure (mmHg): "))
            skin_thickness= int(input("Enter your skin thickness (mm): "))
            Insulin= int(input("Enter your insulin level (IU/ml): "))
            BMI= int(input("Enter your Body Mass Index (kg/m): "))
            Diabetes_pedigree_function= float(input("Enter number of diabetes pedigree function: "))
            Age= int(input("Enter number of your age (years): "))

            input_list=np.array([[pregnancies, glucose, blood_pressure, skin_thickness, Insulin, BMI, Diabetes_pedigree_function, Age]])
            classifier= pickle.load(open(path+"diabetes_model.pkl", 'rb'))
            scalar= pickle.load(open(path+"diabetes_scaler.pkl", 'rb'))
            prediction= classifier.predict(scalar.transform(input_list))

            if prediction==1:
                print("--"*60,sep='\n')
                print(("You have diabetes, Please take care!!!").center(120))
                print("--"*60,sep='\n')
            else:
                print("--"*60,sep='\n')
                print(("Hurray, You don't have diabetes!!!").center(120))
                print("--"*60,sep='\n')
            continue

        if predictor==3: # Malaria
            print("**"*60)
            print(("Malaria Predictions").center(120))
            print("**"*60)
            
            model=tensorflow.keras.models.load_model(path+"Malaria_model.h5")
            print("The model is loaded")
            print("\n")
            print("Please enter a valid path for the cell image: ")
            print("\n")
            image_path=input("Path: ")

            try:
                print("Loading the image")             
                img = Image.open(image_path)
            except:
                print("Invalid file path! ")
                continue

            img.show() 
            
            print("Predicting the image")
            print("\n")
            img=image.load_img(image_path,target_size=(64,64))
            x=image.img_to_array(img)
            x=x/255
            image_1 = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            prediction=model.predict(image_1)[0]
            print("Prediction is completed")
            print("\n")
            p = np.argmax(prediction)
            if p==1:
                print("--"*60,sep='\n')
                print(("Hurray, This cell is uninfected!!!").center(120))
                print("--"*60,sep='\n')
            else:
                print("--"*60,sep='\n')
                print(("This cell is parasitized!!!").center(120))
                print("--"*60,sep='\n')
            continue

        if predictor==4: # Thyroid

            print("**"*60)
            print(("Thyroid prediction").center(100))
            print("**"*60)
            print("\n")
            print("Please Enter the following information: ")

            t3= int(input("Enter T3RESIN value: "))
            tx= float(input("Enter Thyroxin value"))
            tn= float(input("Enter Thyronine value"))
            tv= float(input("Enter Thyroid value"))
            tsh= float(input("Enter TSH value"))

            input_list=np.array([[t3,tx,tn,tv,tsh]])
            classifier= pickle.load(open(path+"finalized_model_thyroid.pkl", 'rb'))
            scalar= pickle.load(open(path+"thyroid_scalar.pkl", 'rb'))
            prediction= classifier.predict(scalar.transform(input_list))

            if prediction==1:
                print("--"*60,sep='\n')
                print(("Hurray, You're Safe!!").center(120))
                print("--"*60,sep='\n')
            elif prediction==2:
                print("--"*60,sep='\n')
                print(("You're at the risk of Hyperthyroidism!! Please get medical assistance").center(120))
                print("--"*60,sep='\n')
            elif prediction==3:
                print("--"*60,sep='\n')
                print(("You're at the risk of Hypothyroidism!! Please get medical assistance").center(120))
                print("--"*60,sep='\n')
            continue

        if predictor == 5: # Pneumonia
            print("**"*60)
            print(("Pneumonia predictions").center(120))
            print("**"*60)
            
            model=tensorflow.keras.models.load_model(path+"Pnuemonia_model.h5")
            print("The model is loaded")
            print("\n")
            print("Please enter a valid path for the Xray image: ")
            print("\n")
            image_path=input("Path: ")

            try:
                print("Loading the image")             
                img = Image.open(image_path)
            except:
                print("Invalid file path! ")
                continue

            img.show() 
            
            print("Predicting the image")
            print("\n")
            img=image.load_img(image_path,target_size=(224,224))
            x=image.img_to_array(img)
            x=x/255
            image_1 = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            prediction=model.predict(image_1)
            print("Prediction is completed")
            print("\n")
            p = np.argmax(prediction[0])
            if p==1:
                print("--"*60,sep='\n')
                print(("Hurray, The result is negative!!!").center(120))
                print("--"*60,sep='\n')
            else:
                print("--"*60,sep='\n')
                print(("You might have Pneumonia! Please visit the nearest Healthcare facility").center(120))
                print("--"*60,sep='\n')
            
            continue
        
        if predictor == 6: # Insect bite
            
            print("**"*60)
            print(("Insect bite classification").center(120))
            print("**"*60)
            model=tensorflow.keras.models.load_model(path+"insect.h5") 
            print("The model is loaded")
            print("\n")
            print("Please enter a valid path for the image: ")
            print("\n")
            image_path=input("Path: ")

            try:
                print("Loading the image")             
                img = Image.open(image_path)
            except:
                print("Invalid file path! ")
                continue

            img.show() 
            
            print("Predicting the image")
            print("\n")
            img=image.load_img(image_path,target_size=(256,256)) 
            x=image.img_to_array(img)
            x=x/255
            image_1 = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            prediction=model.predict(image_1)

            bites = ['mosquito bite','tick']

            print("Prediction is completed")
            print("\n")

            p = np.argmax(prediction[0])
            print("It is a", bites[p])
            continue

        if predictor == 7: # COVID
            
            print("**"*60)
            print(("COVID identification using cough sounds").center(120))
            print("**"*60)
            model=tensorflow.keras.models.load_model(path+"Covid_model.h5") ###############
            print("The model is loaded")
            print("\n")
            print("Please enter a valid path for the audio file: ")
            print("\n")
            fname=input("Path: ")

            try:
                print("Loading the audio file")             
                samples, sample_rate = librosa.load(fname,sr=44100)
                sample = np.array(samples)
                sample = sample.reshape(1,sample.shape[0])
            except:
                print("Invalid file path! ")
                continue

            fig = plt.figure(figsize=[4,4])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            S = librosa.feature.melspectrogram(y=sample, sr=sample_rate)
            #librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            direc = 'spectrogram.jpg'
            plt.savefig(direc)

            img = Image.open(direc)
            img.show() 

            print("Predicting the audio spectrogram")
            print("\n")
            img=image.load_img(direc,target_size=(224,224)) #########################
            x=image.img_to_array(img)
            x=x/255
            image_1 = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
            prediction=model.predict(image_1)
            print("Prediction is completed")
            print("\n")

            
            p = np.argmax(prediction[0])
            if p==1:
                print("--"*60,sep='\n')
                print(("Hurray, The result is negative!!!").center(120))
                print("--"*60,sep='\n')
            else:
                print("--"*60,sep='\n')
                print(("You might have COVID19! Please visit the nearest testing facility").center(120))
                print("--"*60,sep='\n')
            
            continue
          
        