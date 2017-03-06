if __name__ == "__main__":
    import subprocess
    import os
    cmd_api = "sphinx-apidoc -f -o docs/apis OpenGoddard"
    cmd_cd = "cd docs"
    cmd_make = "make html"

    result = subprocess.run(cmd_api, shell=True, universal_newlines=True)
    os.chdir("docs")
    result = subprocess.run(cmd_make, shell=True, universal_newlines=True)
    os.chdir("..")
