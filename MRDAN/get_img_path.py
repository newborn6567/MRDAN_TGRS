import os.path as osp
import os

base_path = "../data/cro-scene/domain_adaptation_images/"
base_path2 = "../data/cro-scene/"

a = os.listdir(base_path)#[AID,RSS,UCM,WHU]
AID_file = open(osp.join(base_path2, "AID_list_test.txt"), "w")
RSSCN7_file = open(osp.join(base_path2, "RSSCN7_list_test.txt"), "w")
num_idx =0
for _domain_adaptation_images_path in a:
    A = os.path.join(base_path,_domain_adaptation_images_path)
    A1 = os.listdir(A)#[images]
    for B0 in A1:
        B= os.path.join(A, B0)
        B1 = os.listdir(B)
        for i, C0 in enumerate(B1):
            C = os.path.join(B, C0)
            C1 = os.listdir(C)
            for D0 in C1:
                if (num_idx % 3 ==0):
                    D = os.path.join( C, D0)
                    idx = str(i)
                    DD = D[8:] + ' ' + idx + '\n'
                    if _domain_adaptation_images_path=="AID":
                        AID_file.write(DD)
                        AID_file.flush()
                    elif _domain_adaptation_images_path=="RSSCN7":
                        RSSCN7_file.write(DD)
                        RSSCN7_file.flush()
                num_idx +=1
AID_file.close()
RSSCN7_file.close()



