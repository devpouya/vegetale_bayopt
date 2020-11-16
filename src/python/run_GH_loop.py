#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:17:57 2020

Reduced function that will run on Grasshoper, that will receive geometries and
output labels. It is almost the same as the script run_generate_batches_from_ml,
implemented by Ania Apolinarska

@author: luissalamanca
"""

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

#dir_path    = os.path.dirname(os.path.realpath(__file__))
dir_path = '/Users/pouya/vegetale_bayopt'

docServer = Grasshopper.GH_InstanceServer.DocumentServer
doc = docServer[0] # first opened document

# Receive message from optimizer about when its done


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


#find all batteries to control
gen_mode_battery = FindBatteryByNickname(doc.Objects, "Generator_Mode")
input_battery = FindBatteryByNickname(doc.Objects, "Externally_set_W")
output_battery = FindBatteryByNickname(doc.Objects, "Output_All_Data")
control_toggle = FindBatteryByNickname(doc.Objects, "Automatic_Control")
control_toggle.Value = True #set GH script to automatic control mode

"""
We will run the loop exchanging the files with the following names:
- tmp_fromscript.pkl : contains the geometries that GH is going to be using, to
    generate the corresponding design attributes
- tmp_fromgh.pkl : the file saved by GH, with the design attributes, i.e., labels

Every time a file is read, it is removed, to not trigger again its reading
"""

not_end = 1
output_default = []
max_iter = 50
iter = 0

while not_end:
    if iter >= max_iter:
        not_end = 0
        break
    iter += 1
    print('Start method')
    t0 = datetime.datetime.now()
    # read a batch of geometry input data predicted by ML
    f_name_out = os.path.join(dir_path,'tmp_fromscript.pkl')
    input_batch_data = wait_gh_x(f_name_out, 3600)
    if input_batch_data is None:
        break
    print('Read file')
    time_start = time.time()

    k = 0
    set_value(gen_mode_battery, 1) #set to re-generator mode

    # run geometry from predicted inputs, one by one
    output_batch_data = []
    for d in input_batch_data:
        input_val = d
        set_value(input_battery, input_val)
        #output = output_battery.VolatileData[0][0].Value
        try:
            output = output_battery.VolatileData[0][0].Value
            print(output)
            #output['geo']=d['geo']
        except:
            output = dict()
        output_batch_data.append(output)
        k+=1

    # save the actual design attributes (scores)
    output_file = 'tmp_fromgh.pkl'
    with open(os.path.join(dir_path, output_file), 'wb') as f:
        pickle.dump(output_batch_data, f, protocol=2)

    flag_save = input_batch_data[0]['flag_save']
    if len(flag_save):
        output_file = 'saved/tmp_fromgh_{}.pkl'.format(flag_save)
        with open(os.path.join(dir_path, output_file), 'wb') as f:
           pickle.dump(output_batch_data, f, protocol=2)

    control_toggle.Value = False #set GH script to back to manual control mode
    end_time = time.time() - time_start
    #if os.path.exists(dir_path+"finished.txt"):
    #    not_end = 0
    print('Run GH for %d samples in total time of %.2f'%(len(input_batch_data),end_time))

    #--- save some stats------------------------------------------------------------
    t1 = datetime.datetime.now()
    with open(os.path.join(dir_path,'info.txt'),'w') as f:
        f.write("t0=%s\n"%t0)
        f.write("t1=%s\n"%t1)
        f.write("dt=%s\n"%(t1-t0))
