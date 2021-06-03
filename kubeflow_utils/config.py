
from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="KUBEFLOW",
    settings_files=['settings.yaml'],
)

# `envvar_prefix` = export envvars with `export KUBEFLOW_FOO=bar`.
# `settings_files` = Load this files in the order.
