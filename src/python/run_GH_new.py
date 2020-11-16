import os
import pickle
import time, datetime

try:
    import Rhino
    Grasshopper = Rhino.RhinoApp.GetPlugInObject("Grasshopper")
    import Grasshopper
except:
    #because the first try sometimes doesn't work after "Reset Script Engine"
    import clr
    clr.AddReference('Grasshopper')
    import Grasshopper

dir_path    = os.path.dirname(os.path.realpath(__file__))
dir_path = '/Users/pouya/vegetale_bayopt'
output_path  = os.path.join(dir_path,'generated')

docServer = Grasshopper.GH_InstanceServer.DocumentServer
doc = docServer[0] # first opened document


def FindBatteryByNickname(docObjects, name):
    if docObjects is None: return None

    for obj in docObjects:
        attr = obj.Attributes

        if attr.PathName == name:
            return obj
    raise Exception(name + " was not found in document")

def set_value(battery, val):
    battery.Script_ClearPersistentData()
    battery.AddPersistentData(val)
    battery.ExpireSolution(True)

def wait_gh_x(f_name_out, max_time = 3600 * 1):
    # Wait for file and open the content
    in_time = time.time()
    # We first convert them to the format required by GH
    not_end = 1
    while not_end:
        if os.path.exists(f_name_out):
            check_size = 1
            prev_size = -1
            while check_size:
                if prev_size == os.path.getsize(f_name_out):
                    check_size = 0
                else:
                    prev_size = os.path.getsize(f_name_out)
                    time.sleep(2)
            dict_polar = pickle.load(open(f_name_out,'rb'))
            os.remove(f_name_out)
            return dict_polar
        else:
            time.sleep(5)
            if (time.time() - in_time) > max_time:
                print('Max time reached!')
                return None

# === do it once in the beginning, before sending/fetching samples  ===

#find all batteries to control
mode_battery = FindBatteryByNickname(doc.Objects, "Generator_Mode")
input_battery = FindBatteryByNickname(doc.Objects, "Externally_set_W")
output_battery = FindBatteryByNickname(doc.Objects, "Output_All_Data")
control_toggle = FindBatteryByNickname(doc.Objects, "Automatic_Control")

control_toggle.Value = True #set GH script to automatic control mode
control_toggle.ExpireSolution(True)
set_value(mode_battery, 3) #set to explicit input mode

ind_f = '0'

# === you might want to loop this part ===
f_name_out = os.path.join(dir_path,'tmp_fromscript.pkl')
for _ in range(1):
    #input_batch_data = pickle.load(open(f_name_out,'rb'))
    input_batch_data = wait_gh_x(f_name_out, 3600)


    # run geometry from predicted inputs, one by one
    output_batch_data = []
    #pickle.dump(f_name_out, open('{}/file_test_start.pkl'.format(dir_path), 'wb'), protocol=2)
    for idb, d in enumerate(input_batch_data):
        input_val = [d]
        set_value(input_battery, input_val)
        #output = output_battery.VolatileData[0][0].Value
        try:
            output = output_battery.VolatileData[0][0].Value
            #output['geo']=d['geo']
        except:
            #pickle.dump(idb, open('{}/file_test_{}.pkl'.format(dir_path,idb), 'wb'), protocol=2)
            output = dict()
            print("check for errors in GH - no data in output component") #won't print, some Rhino bug
        output_batch_data.append(output)

    output_file = 'tmp_fromgh.pkl'.format(ind_f)
    with open(os.path.join(dir_path, output_file), 'wb') as f:
        pickle.dump(output_batch_data, f, protocol=2)

# === at the end, after all loops finished ===

control_toggle.Value = False #set GH script to back to manual control mode
control_toggle.ExpireSolution(True)
