import os

root_path = "C:/Users/Scott/OneDrive/Desktop/vaporize_theia/"
# get the model names by reading the subdirectories in the root path
model_names = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]

# loop through the models and open the ejecta_composition.txt file and theia_composition.txt file
for model_name in model_names:
    print(f"At model {model_name}")
    path = root_path + model_name + "/" + model_name + "/"
    ejecta_composition = eval(open(path + "ejecta_composition.csv", "r").read())
    theia_composition = eval(open(path + "theia_composition.csv", "r").read())
    print(ejecta_composition.keys())
