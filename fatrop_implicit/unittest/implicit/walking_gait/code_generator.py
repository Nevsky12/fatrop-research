from casadi import *
import os

def get_code_generated_function(f, save=True):
    compiler = 'gcc'
    name = f.name()
    soname = f'{name}.so'
    cname = f'{name}.c'
    folder = 'code_generation'
    
    # Cleanup old generated files
    for file in [soname, cname, f'{name}.o']:
        if os.path.exists(f"{folder}/{file}"):
            os.remove(f"{folder}/{file}")
        
    f.generate(cname)
    # Compile the C code
    fatrop_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '/usr/local/lib') 
    command = f'{compiler} -fPIC -shared {cname} -o {soname} -L{fatrop_lib_path} -Wl,-rpath,{fatrop_lib_path} -lfatrop'
    os.system(command)

    # Load the shared library   
    f_new = external(f.name(), soname)
    
    # Cleanup .c and .so files after generation
    if os.path.exists(cname):
        os.remove(cname)
    if os.path.exists(soname):
        os.remove(soname)
    
    # save this function for later use
    if save:
        f_new.save(f"casadi_funcs/{f.name()}_code_generated.casadi")
    
    return f_new