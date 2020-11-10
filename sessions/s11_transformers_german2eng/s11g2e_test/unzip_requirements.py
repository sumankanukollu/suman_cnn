import os
import shutil
import sys
import zipfile


pkgdir = '/tmp/pkgs-from-layer'

# insert the pkgdir into system path
sys.path.insert(1, pkgdir)

req_pkg_path= '/tmp/pkgs-from-layer/requirements'

sys.path.insert(1,req_pkg_path)

if not os.path.exists(pkgdir):
    tempdir = '/tmp/_pkgs-from-layer'
    if os.path.exists(tempdir):
        shutil.rmtree(tempdir)

    layer_copy_root = '/opt'
    zip_requirements = os.path.join(layer_copy_root, 'requirements.zip')

    zipfile.ZipFile(zip_requirements, 'r').extractall(tempdir)
    os.rename(tempdir, pkgdir)  # Atomic

