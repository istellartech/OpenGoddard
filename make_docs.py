if __name__ == "__main__":
    import subprocess
    import os
    cmd_api = "sphinx-apidoc -f -o docs_sphinx/apis OpenGoddard"
    cmd_cd = "cd docs_sphinx"
    cmd_make = "make html"

    result = subprocess.run(cmd_api, shell=True, universal_newlines=True)
    os.chdir("docs_sphinx")
    result = subprocess.run(cmd_make, shell=True, universal_newlines=True)
    os.chdir("..")
