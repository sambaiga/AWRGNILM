from lilac.load_data import *
from lilac.visualizer_functions import *
from lilac.transform_functions import *
from lilac.functions import *
from lilac.load_data import *
from lilac.processing_functions import *
import random
from tqdm import tqdm


events = {"1": ([10, 15, 20], [40, 45, 50]),
          "2": ([10, 40], [160, 190]),
          "3": ([10, 15], [45, 50]),
          "4": ([10, 20], [50, 60]),
          "5": ([10, 40], [220, 250]),
          "6": ([10, 20], [80, 90]),
          "7": ([10, 20], [50, 80]),
          "8": ([10, 110], [190, 220]),
          "9": ([10, 40], [190, 220]),
          "10": ([10, 20], [50, 60]),
          "11": ([10, 15], [75, 80]),
          "12":  ([10, 15], [40, 45])
          }


def get_data(full_path, file_id):

    file_name = os.path.basename(full_path)
    names = file_name.strip().split(".")[0].strip().split("_")
    appliances = names[:-1]

    on_pattern = names[-1]
    on_pattern = [int(x) for x in on_pattern]
    if len(on_pattern) == 3:
        on_pattern = on_pattern * 2

    readings = read_tdms(full_path)
    df = pd.DataFrame(columns=['I1', 'V1', 'I2', 'V2', 'I3', 'V3']+appliances)

    for k in range(1, 4):

        df['I{}'.format(k)] = readings['I{}'.format(k)]
        df['V{}'.format(k)] = readings['V{}'.format(k)]

    t = np.arange(len(df))*2
    on_time = [x*1e5 for x in events[file_id][0]]
    off_time = [x*1e5 for x in events[file_id][1]]

    slices = []
    apps = []
    windows_on = []
    windows_off = []
    for i in range(len(appliances)):
        slices.append(slice(np.where(t == on_time[i])[
            0][0], np.where(t == off_time[i])[0][0]))
        apps.append(np.zeros_like(t))
        windows_on.append(np.where(t == on_time[i])[0][0])
        windows_off.append(np.where(t == off_time[i])[0][0])

    for i in range(len(appliances)):
        apps[i][slices[i]] = 1

    on_events = np.zeros_like(t)

    for window in windows_on:
        on_events[window] = 1

    for window in windows_off:
        on_events[window] = -1

    df["on-events"] = on_events

    return df, appliances,  on_pattern


def get_activations(full_path, file_id, data_filter=True,  trans=None, transform_type="ct", num_cycles=5, verbose=False, vis=True, three_phases=True):

    period = int(50e3/50)
    cycles = period*num_cycles
    current_appliance_id = []
    voltage_appliance_id = []

    final_currents = []
    final_voltages = []
    final_power = []
    i_max = []

    df, appliances,  on_pattern = get_data(full_path, file_id)

    events = df[(df["on-events"] == 1) | (df["on-events"] == -1)]["on-events"]
    df = df[["I1", "I2", "I3", "V1", "V2", "V3"]]
    events_appliance_id = events.index.values
    events_value = np.where(events.values == 1, 1, 0).tolist()
    apps_name = [appliances[on_pattern[i]-1] for i in range(len(on_pattern))]
    apps_name = appliances_check(apps_name)

    labels = apps_name

    #power = get_three_phase_power(df)

    if data_filter:
        if verbose:
            print("filter data")

        def filterF(x): return butter_lowpass_filter(x.values, nyq=0.005)
        df = df.apply(filterF)
    voltages = ["V1", "V2", "V3"]
    currents = ["I1", "I2", "I3"]
    c = df[currents].values
    v = df[voltages].values
    """
    p = power.values
    power_ids = [int(events_appliance_id[i]/1000)
                 for i in range(len(events_appliance_id))]
    """
    current_appliance_id += [c[events_appliance_id[idx]-cycles:events_appliance_id[idx]+cycles]
                             for idx in range(len(events_appliance_id))]
    voltage_appliance_id += [v[events_appliance_id[idx]-cycles:events_appliance_id[idx]+cycles]
                             for idx in range(len(events_appliance_id))]
    on_events = events_value

    if vis:
        ev = np.zeros(len(df))
        ev[events_appliance_id[0]] = 9
        ev[events_appliance_id[1]] = 8
        ev[events_appliance_id[2]] = 5
        ev[events_appliance_id[3]] = 5
        if len(labels) > 4:
            ev[events_appliance_id[4]] = 5
            ev[events_appliance_id[5]] = 5
        plt.plot(c[:, 0])
        plt.plot(ev)
        plt.text(events_appliance_id[0], 9, labels[0])
        plt.text(events_appliance_id[1], 8, labels[1])
        plt.text(events_appliance_id[2], 5, labels[2])
        plt.text(events_appliance_id[3], 5, labels[3])
        if len(labels) > 4:
            plt.text(events_appliance_id[4], 4, labels[4])
            plt.text(events_appliance_id[5], 4, labels[5])
        plt.show()

    for idx in range(len(on_events)):
        c = current_appliance_id[idx]
        v = voltage_appliance_id[idx]
        
        
        ##get power
        """
        if idx==0:
            power=p[power_ids[idx]-cycles:power_ids[idx]]
        elif idx==len(on_events)-1:
            power=p[power_ids[idx]:power_ids[idx]+cycles]
        else:
            power=p[power_ids[idx-1]:power_ids[idx]]
                    
        final_power+=[power]
        """

        

    
        if trans is None:
            I, V = get_three_phase_VI(c, v, events_value[idx])
            Imax = max(get_max_current_from_list(I))
            i_max += [Imax]
            final_currents += [I]
            final_voltages += [V]
            
            if vis:
                plot_VI_from_list(I, V, Imax)
                plt.title(f'{labels[idx]}: orgn', fontsize=10)
                plt.show()
            
        

        elif trans == 0:
            I, V = get_three_phase_VI(c, v, events_value[idx])
            if transform_type == "ct":
                i_ct = CT_transform(np.vstack(I).T)
                v_ct = CT_transform(np.vstack(V).T)
            elif transform_type == "isc":
                i_ct = isc_transform(np.vstack(I).T)
                v_ct = isc_transform(np.vstack(V).T)
                
            Imax = max(get_max_current_from_list(i_ct))
            i_max += [Imax]

            vt_id = np.where(v_ct.max(0) == v_ct.max())[0][0]
            ct_id = np.where(i_ct.max(0) == i_ct.max())[0][0]

            final_currents += [i_ct[:, ct_id]]
            final_voltages += [V[0]]

            if vis:
                for k in range(0, 3):
                    plt.subplot(1, 3, k+1)
                    plt.plot(i_ct[:, k])
                    plt.ylim(-Imax, Imax)
                plt.title(f'{labels[idx]}: phase={k}', fontsize=10)
                plt.tight_layout()
                plt.show()
                plt.plot((V[0]+V[1]+V[2])*1/3, i_ct[:, ct_id])
                plt.title(
                    f'{labels[idx]}: state={events_value[idx]}', fontsize=10)
                plt.show()

        elif trans == 1:
            Ie, Ve, imb = get_transform(c,v, events_value[idx], transform_type)
            Imax = max(get_max_current_from_list(Ie))
            i_max += [Imax]
            

            if three_phases:
                final_currents += [Ie]
                final_voltages += [Ve]
            else:
                if imb[0] > imb[1]:
                    columns = [0, 2]
                elif imb[0] < imb[1] and imb[0] < 0.5:
                    columns = [0, 1]
                else:
                    columns = [0, 2]

                Ie = [Ie[index] for index in columns]
                Ve = [Ve[index] for index in columns]

                final_currents += [Ie]
                final_voltages += [Ve]
                

            Iemax = get_max_current_from_list(Ie)
            max_ratio = np.round(Imax/np.array(Iemax), 2)

            if vis:
                ilim = max(Iemax)
                #plot_current_from_list(Ie, Ve, ilim)
                #plt.show()
                plot_VI_from_list(Ie, Ve, ilim)
                plt.title(f'{labels[idx]}:{imb[0]}:{imb[1]}', fontsize=8)
                plt.show()
                
        elif trans==2:
            Ie, Ve = get_adaptive_transform(c,v, events_value[idx], transform_type)
            Imax = max(get_max_current_from_list(Ie))
            i_max += [Imax]
            final_currents += [Ie]
            final_voltages += [Ve]
            if vis:
                ilim = Imax
                #plot_current_from_list(Ie, Ve, ilim)
                #plt.show()
                plot_VI_from_list(Ie, Ve, ilim)
                plt.title(f'{labels[idx]}', fontsize=8)
                plt.show()

    if verbose:
        print(f"on_pattern:{on_events}: size:{len(on_events)}")
        print(f"labels:{labels}: size:{len(labels)}")
        #print(f"power_id:{power_ids}: size:{len(power_ids)}")
        print(f"current:{len(final_currents)}")
        print(f"voltage:{len(final_voltages)}")
        print(f"max_current:{len(i_max)}")

    return final_currents, final_voltages, labels, on_events,  i_max


def select_aggregate_data_appliance_type(path,  data_filter=False,  trans=2, transform_type="ct", num_cycles=10, verbose=False, vis=False):

    current = []
    voltage = []
    power = []
    labels = []
    states = []
    max_current = []
    power_events = []

    files = getListOfFiles(path)
    print(len(files))

    print("Load data")
    files_id = 0
    with tqdm(total=len(files)) as pbar:
        for root, k, fnames in sorted(os.walk(path)):
            file_id = root.strip().split("/")[-1]
            if file_id:
                for j, fname in enumerate(sorted(fnames)):
                    full_path = os.path.join(root, fname)
                    #print(fname)
                    c, v, l, s, c_max = \
                        get_activations(full_path, file_id, data_filter,
                                        trans, transform_type, num_cycles, verbose, vis)
                    current += c
                    voltage += v
                    labels += l
                    states += s
                    max_current += c_max
                    #power+=p
                    #power_events+=pid
                    pbar.set_description('processed: %d' % (1 + files_id))
                    pbar.update(1)
                    files_id += 1
        pbar.close()
    print(f"currents size:{len(current)}")  
    print(f"labels size:{len(labels)}")
    print(f"states:{len(states)}") 
    print(f"voltage:{len(voltage)}") 
    print(f"max_current:{len(max_current)}")
    #print(f"power:{len(power)}") 
    #print(f"power_events:{len(power_events)}") 
    assert len(current)==len(voltage)==len(labels)
         
    return np.array(current), np.array(voltage), np.array(labels), np.array(states), np.array(max_current)


def generate_data():
    path = "../../data/Triple/"
    save_path="../data/lilac/aggregated/"
    current, voltage, labels, states, max_current = select_aggregate_data_appliance_type(path,  data_filter=False,  trans=None, 
                                                     transform_type=None, num_cycles=10, verbose=False, vis=False)
    np.save(save_path+"current.npy", current)
    np.save(save_path+"voltage.npy", voltage)
    np.save(save_path+"labels.npy", labels)
    np.save(save_path+"states.npy", states)


def generate_isc_data():
    path = "../../data/Triple/"
    save_path="../data/lilac/aggregated_isc/"
    current, voltage, labels, states, max_current = select_aggregate_data_appliance_type(path,  data_filter=False,  trans=1, 
                                                     transform_type="isc", num_cycles=10, verbose=False, vis=False)
    np.save(save_path+"current.npy", current)
    np.save(save_path+"voltage.npy", voltage)
    np.save(save_path+"labels.npy", labels)
    np.save(save_path+"states.npy", states)


if __name__ == "__main__":
    generate_isc_data()
    
