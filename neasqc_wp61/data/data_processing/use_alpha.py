import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_path + "../../models/quantum/alpha/module/")
import json
import numpy as np
import dataset_wrapper 
import parameterised_quantum_circuit
import alpha_trainer 
import alpha_model 
##################################################################################################################################################################
##################################################################################################################################################################
##################################################################################################################################################################

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", help = "Seed for the initial parameters")
    parser.add_argument("-e", "--epochs", help = "Number of epochs of training over the dataset")
    parser.add_argument("-r", "--runs", help = "Number of runs of the training(results are averaged over this)")
    parser.add_argument("-tr", "--train", help = "Directory of the train dataset")
    parser.add_argument("-te", "--test", help = "Directory of the test datset")
    parser.add_argument("-o", "--output", help = "Output file with the predictions")
    args = parser.parse_args()


    
    ########################################
    #How many generations(epochs) to be ran?
    number_of_epochs = args.epochs
    # Run the training number_of_runs times and average over the results
    number_of_runs = args.runs
    #Set random seed
    seed = args.seed
    ########################################
    ###Training the model
    
    
    
    print("Initialisation Begun \n")
    trainer = alpha_trainer(args.train, seed)
    print("Initialisation Ended \n")

    for i in range(number_of_runs):
        print("run = ", i+1, "\n")
        if i==0:
            loss_array, accuracy_array, prediction_array = trainer.train(number_of_epochs)
        else:
            loss_temp_array, accuracy_temp_array, prediction_temp_array  = trainer.train(number_of_epochs)
            loss_array += loss_temp_array
            accuracy_array += accuracy_temp_array
            prediction_array += prediction_temp_array 
    loss_array = loss_array/number_of_runs
    accuracy_array = accuracy_array/number_of_runs
    prediction_array = prediction_array/number_of_runs

    save_array = list(prediction_array)

    #Save save_array to results/raw           
    file = open(args.output, "w")
    for item in save_array:
        file.write(f"{int(item)}"+"\n")
    file.close()  

if __name__ == "__main__":
    main()              