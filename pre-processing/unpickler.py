import pickle
import os

"""
Unpickle using pickle package for reliability. Converted to .csv
"""

rootdir = r"data\chambers-2020\infant_movement_assessment_repo_files"
print(rootdir+r"\data\pose_estimates\clinical\py\pose_estimates.pkl")
try:
    os.rename(rootdir+r"\data\pose_estimates\clinical\py\pose_estimates.pkl", 
          rootdir+r"\data\pose_estimates\clinical\py\pose_estimates_clin.pkl")
except FileNotFoundError:
    print("File not found: either already renamed, or inexistent.")
    quit()

for root, dirs, files in os.walk(rootdir):
    for filename in files:
        if filename.endswith('clin.pkl'):
            path = os.path.join(root,filename)
            with open(path, 'rb') as file:
                unpickled = pickle.load(file)
                unpickled.to_csv(f"pre-processing/additional-files/{filename.split('.')[0]}.csv")
                print(path+" unpickled to csv.")
print("Unpickling succesfully ended.")