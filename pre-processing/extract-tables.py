import pandas as pd
import numpy as np
import warnings
import ast
import os

########################################################
#CHAMBERS DATA TABLE

#import csvs
warnings.simplefilter(action='ignore',category=FutureWarning)
meta_data_clinical = pd.read_csv(r"pre-processing\additional-files\meta_data_clin.csv", index_col="Unnamed: 0")
pose_clinical = pd.read_csv(r"pre-processing\additional-files\pose_estimates_clin.csv", index_col="Unnamed: 0")

#create sorter for body-part order standard
pose_clinical = pose_clinical.drop(['fps','part_idx','time'], axis=1)
with open(r"pre-processing\additional-files\chambers-pose-labels.txt") as bp_order:
    sorter = [l.split(' ')[1].strip('\n') for l in bp_order.readlines()]

#create list of individual videos
list_of_inf_vids = []
for infant in pose_clinical['infant'].unique():
    curr_inf = pose_clinical[pose_clinical['infant'] == infant]
    curr_n_vids = curr_inf['video'].unique()
    for vid in range(len(curr_n_vids)):
        list_of_inf_vids.append(curr_inf[curr_inf['video'] == curr_n_vids[vid]])

#create list of videos ordered by the body-part standard order and frame order
ordered_list_of_inf_vids = []
for inf_vid in list_of_inf_vids:
    temp_categorical_order = pd.Categorical(inf_vid['bp'], categories=sorter, ordered=True)
    inf_vid['temp_cat_ord'] = temp_categorical_order
    inf_vid = inf_vid.sort_values(['temp_cat_ord', 'frame'])
    inf_vid = inf_vid.drop('temp_cat_ord', axis=1)
    ordered_list_of_inf_vids.append(inf_vid)

#start formatting for the final table: (i) create a frame-lengthed list of the 18 available 2D coordinates; (ii) each row of the resulting table is a unique video, 
#frame-lengthed coordinate list and a label, in accordance with the metadata.
chambers_data = pd.DataFrame(columns=['video', 'infant', 'coordinates', 'label'])
for i in range(len(ordered_list_of_inf_vids)):
    vid = ordered_list_of_inf_vids[i]
    vid = vid.reset_index()
    frame_lengthed_list_of_lists = []
    for j in vid['frame'].unique():
        frame_lengthed_list_of_lists.append([[x,y] for x,y in zip(vid[vid['frame']==j]['x'], vid[vid['frame']==i]['y'])])
    chambers_data = chambers_data.append({'video':i,
                               'infant':vid['infant'].unique()[0],
                               'coordinates':frame_lengthed_list_of_lists,
                               'label':np.nan}, 
                                ignore_index=True)

#join metadata and pose tables and add a 'label' column corresponding to the metadata risk corr/chron columns.
chambers_data = chambers_data.join(meta_data_clinical.iloc[:,1:-1])
chambers_data['label'] = chambers_data['Risk_low0_mod1_high2_chron'].fillna(0).astype(int)
chambers_data['label'] = chambers_data['Risk_low0_mod1_high2_corr'].where(pd.notnull(chambers_data['Risk_low0_mod1_high2_corr']), chambers_data['label']).astype(int)
chambers_data = chambers_data.set_index('video')
#chambers_data = chambers_data.drop(columns=['video'])

########################################################
#GONG DATA TABLE

#filepaths:
labels_path = r"data/gong-2022/pmi-gma/joint_points/joint_points.txt"

#id/motion label table:
label_table = pd.read_csv(labels_path, sep=' ', index_col=0)
label_table.index.name = "child_id"
label_table.columns = ["label"]
label_table.sort_index(inplace=True)
label_table['ID'] = label_table.index.to_series().apply(lambda x: x.split('_')[0]).map(int)
label_table.drop_duplicates(subset=['ID'], inplace=True, keep='last')
label_table['full_child_id'] = label_table.index
label_table.index = label_table['ID']
label_table.drop(['ID'], axis=1, inplace=True)

#id/joint_coordinates table:
dir_path_0 = r"data/gong-2022/pmi-gma/joint_points/411"
dir_path_1 = r"data/gong-2022/pmi-gma/joint_points/709"
file_list_0 = os.listdir(dir_path_0)
file_list_1 = os.listdir(dir_path_1)
frame_coord_list = []

#populate lists
for joint_list_name in file_list_0:
    with open(dir_path_0+"\\"+joint_list_name) as joint_list:
        joint_list = joint_list.readlines()
        joint_list = [ast.literal_eval(coord) for coord in joint_list]
        frame_coord_list.append((joint_list_name.strip('.txt'), joint_list))
for joint_list_name in file_list_1:
    with open(dir_path_1+"\\"+joint_list_name) as joint_list:
        joint_list = joint_list.readlines()
        joint_list = [ast.literal_eval(coord) for coord in joint_list]
        frame_coord_list.append((joint_list_name.strip('.txt'), joint_list))

gong_data = pd.DataFrame(frame_coord_list, columns=['full_child_id', 'coordinates'])
gong_data.set_index('full_child_id', inplace=True)
gong_data.sort_index(inplace=True)
gong_data['ID'] = gong_data.index.to_series().apply(lambda x: x.split('_')[0]).map(int)
for id_c in gong_data['ID']:
  for id_l in label_table.index:
    if (id_c==(id_l)):
      gong_data.loc[gong_data['ID']==id_c,'label'] = label_table.loc[id_l].label

# coordinates_table contains the unique id for each child, a 0/1 label and a list of 300 lists of 17 lists of 2 floating point numbers, corresponding to the 300 frames, 17 joint labels, and 2D coordinates. 

"""
20231269 is duplicated
20232800 is duplicated
20238794 has _2 ending *only*
20239397 has _1 _2 endings
20239541 has _1 _2 endings
20240065 has _1 _2 endings
20240175 has _1 _2 endings
6 duplicates
"""


#save tables to corresponding data folders.
gong_data.to_csv(r"data/gong-2022/gong-table.csv")
chambers_data.to_csv(r"data/chambers-2020/chambers-table.csv")