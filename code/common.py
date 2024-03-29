import os, os.path, errno, sys, glob, subprocess
import hashlib, pickle, time, multiprocessing, json
import numpy as np
from collections import namedtuple

def mkdir_p(path):
    """Recursively create a directory path"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def is_primitive(val) :
    """ Check if value is a 'primitive' type"""
    primitive_types = [int, float, bool, str]
    return type(val) in primitive_types


def get_basic_conf(conf_module) :
    """
    Get dictionary of values in conf_module, excluding keys starting with '__',
    and include only values with True is_primitive(val)
    """
    d = conf_module.__dict__
    
    # Filter out keys starting with '__',
    # Make sure values are a "primitive" type
    new_dict = {}
    for key, val in d.items() :
        if key[0:2] != "__" and is_primitive(val) :
            new_dict[key] = val
    
    return new_dict


def dict_to_obj(basic_conf_dict) :
    return namedtuple("conf", basic_conf_dict.keys())(*basic_conf_dict.values())


def get_hash(basic_conf) :
    """Convert module -> dictionary with only 'primitive' members -> serialized string -> md5"""
    serialized__str = json.dumps(basic_conf, sort_keys=True).encode('utf-8')
    return hashlib.md5(serialized__str).hexdigest()


def get_output_directory(basic_conf, dir_base = '') :
    basic_conf_id_str = get_hash(basic_conf)
    output_directory = "output/" + basic_conf_id_str + "/" + dir_base + "/"
    mkdir_p(output_directory)
    return output_directory


def dump(basic_conf, data, file_base, dir_base = '') :
    """
    Dump data to a path uniquely determined by a hashed basic_conf
    along with a metadata file containing (primitive) data from basic_conf.
    """
    if not exists(basic_conf, 'metadata') :
        write_basic_conf(basic_conf)

    # Output data
    output_directory = get_output_directory(basic_conf, dir_base)
    filename = output_directory + file_base + '.p'
    pickle.dump( data, open( filename, "wb" ) )


def write_basic_conf(basic_conf) :
    output_directory = get_output_directory(basic_conf)
    print("Writing basic_config in", output_directory)

    # Pickled basic_conf data for re-reading
    filename = output_directory + "metadata.p"
    pickle.dump( basic_conf, open( filename, "wb" ) )
    
    # Human-readable basic_conf data
    filename = output_directory + "metadata.txt"
    fout = open(filename, "w")
    for key, val in basic_conf.items():
        fout.write(str(key) + ' = '+ str(val) + '\n')
    fout.close()


def load_basic_conf(hashstr) :
    """Load basic conf data from a given hash string"""
    filename = "output/" + hashstr + "/metadata.p"
    if not os.path.isfile(filename) :
        raise Exception('Data associated with hash "'+str(hashstr)+'" not found.')
    data = pickle.load( open( filename, "rb" ) )
    return data


def load(basic_conf, file_base, dir_base = '') :
    """Load data from a path uniquely determined by a hashed basic_conf"""
    basic_conf_hash = get_hash(basic_conf)
    output_directory = get_output_directory(basic_conf, dir_base)
    filename = output_directory + file_base + '.p'
    data = pickle.load( open( filename, "rb" ) )
    return data

def timed_load(basic_conf, file_base, dir_base = '') :
    """Load data from a path uniquely determined by a hashed basic_conf"""
    basic_conf_hash = get_hash(basic_conf)
    print("Loading "+file_base+" data. Hash is", basic_conf_hash)
    start = time.time()
    data = load(basic_conf, file_base, dir_base)
    end = time.time()
    print("Done loading "+file_base+" in", end-start, "seconds.")
    return data


def exists(basic_conf, file_base, dir_base = '') :
    """Check if data exists at a path"""
    output_directory = get_output_directory(basic_conf, dir_base)
    filename = output_directory + file_base + '.p'
    return os.path.isfile(filename)


def plots_path(basic_conf, dirname) :
    basic_conf_hash = get_hash(basic_conf)
    # Make plots directory
    plots_path = "output/"+basic_conf_hash+"/plots/" + dirname + "/"
    mkdir_p(plots_path)
    return plots_path


def get_n_cores() :
    cpu_count = multiprocessing.cpu_count()
    if cpu_count > 8 :
        # probably on a cluster node, use the whole thing
        return cpu_count
    else :
        # probably something local, save a couple cores
        return max(1, int( cpu_count - 2 ))
