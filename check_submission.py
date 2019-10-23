#!/usr/bin/python3

import argparse, os, subprocess, shutil


def check_runtime(width, height, d, n, p, path, time_bd, repeat):
  bench = "bench.csv"
  warmup = 1

  time_bd_total = 2 * time_bd * (warmup + repeat)

  if os.path.exists(path + "/diffusion"):
    os.remove(path + "/diffusion")
  os.symlink("/home/hpc2019/a4_grading/test_data/diffusion_{}_{}".format(width,height), path + "/diffusion")

  heat_diffusion_cmd = "/usr/lib64/openmpi/bin/mpirun -n {p} ./heat_diffusion -d{d} -n{n}".format(p = p, d = d, n = n)
  cmd = "hyperfine --export-csv {bench} --time-unit millisecond --warmup {warmup} --max-runs {repeat} \"{cmd}\"".format(bench = "bench.csv", warmup = warmup, repeat = repeat, cmd = heat_diffusion_cmd)
  try:
    subprocess.Popen(cmd, shell = True, cwd=path, stdout = subprocess.DEVNULL, stderr = subprocess.DEVNULL).wait(time_bd_total)
  except subprocess.TimeoutExpired:
    print("TOO SLOW FOR {} ON diffusion_{}_{}".format(heat_diffusion_cmd, width, height))
    return False

  try:
    with open(os.path.join(path, bench), "r") as f:
      time = float(f.readlines()[-1].split(",")[-2])
      # print("CALIBRATE: {} ON diffusion_{}_{} -> {}".format(heat_diffusion_cmd, width, height, 2*time))
      if time > time_bd:
        print("TOO SLOW ({} ms) FOR {} ON diffusion_{}_{}".format(time, heat_diffusion_cmd, width, height))
        return False
      else:
        return True
  except:
    pass

  print("RUNTIME ERROR FOR {} ON diffusion_{}_{}".format(heat_diffusion_cmd, width, height))
  return False

def check_runtimes(path):
  return (   check_runtime( 100   , 100   , "0.01" , 100000 , 1  , path , 1.78 , 10 )
         and check_runtime( 100   , 100   , "0.01" , 100000 , 10 , path , 2.20 , 10 )
         and check_runtime( 10000 , 10000 , "0.02" , 1000   , 2  , path , 97.6 , 3 )
         and check_runtime( 10000 , 10000 , "0.02" , 1000   , 5  , path , 51.3 , 3 )
         and check_runtime( 10000 , 10000 , "0.02" , 1000   , 10 , path , 29.8 , 3 )
         and check_runtime( 10000 , 1000  , "0.6"  , 200    , 1  , path , 4.1  , 3 )
         and check_runtime( 10000 , 1000  , "0.6"  , 200    , 5  , path , 1.7  , 3 )
         )


passed = True

subprocess.run(["mkdir", "-p", "extracted"])


parser = argparse.ArgumentParser()
parser.add_argument("tarfile")
args = parser.parse_args()

tar_file = args.tarfile
assert( tar_file.endswith(".tar.gz") )

stem = tar_file[:-7]
while stem.find("/") != -1:
  stem = stem[stem.find("/")+1:]

extraction_path = "extracted/" + stem
if os.path.isdir(extraction_path):
  print("FATAL: extreaction path {} exists already".format(extraction_path))
  exit(1)


# extract
print( "extracting..." )
subprocess.run(["mkdir", extraction_path])
subprocess.run(["tar", "xf", tar_file, "-C", extraction_path])

# check files
is_valid_file = lambda f: ( 
     f in ["makefile", "Makefile"] or
     f.endswith(".incl") or f.endswith(".cl") or
     f.endswith(".cc") or f.endswith(".c") or
     f.endswith(".hh") or f.endswith(".h") )

print( "checking for additional files..." )
if not all(root == "material" or all(map(is_valid_file, files)) for (root, _, files) in os.walk(extraction_path)):
  print("ADDITIONAL FILES IN TAR")
  passed = False

# build clean build
print( "bulding and cleaning..." )
subprocess.Popen(["make", "heat_diffusion"], cwd=extraction_path).wait()
subprocess.Popen(["make", "clean"], cwd=extraction_path).wait()
if not all(root == "material" or all(map(is_valid_file, files)) for (root, _, files) in os.walk(extraction_path)):
  print("ADDITIONAL FILES AFTER BUILD CLEAN")
  passed = False
subprocess.Popen(["make", "heat_diffusion"], cwd=extraction_path, stdout=subprocess.DEVNULL).wait()


# check times
print( "checking runtimes..." )
passed = passed and check_runtimes(extraction_path)


# clean
print( "final cleaning..." )
shutil.rmtree(extraction_path)


# feedback summary
if passed:
  print("submission passes script")
else:
  print("submission DOES NOT pass script")
