import shutil
import os
import functions
source = os.listdir("Training/Training2/trainingA/")
destination = "testing/"
train_listA, test_listA, train_listB, test_listB = functions.get_train_test()
for file_name in test_listA:
    current_file_name = "p{0:06d}.psv".format(file_name)
    file_to_open = os.path.join("Training/Training2/trainingA/training/", current_file_name)
    shutil.copy(file_to_open,destination)

# =============================================================================
# for file_name in test_listB:
#     current_file_name = "p{0:06d}.psv".format(file_name)
#     file_to_open = os.path.join("Training/Training2/trainingB/training_setB/", current_file_name)
#     shutil.copy(file_to_open,destination)
# =============================================================================
