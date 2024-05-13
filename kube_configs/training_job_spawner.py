import json
import logging
import pathlib
import time

import yaml
from jinja2 import Environment, FileSystemLoader

import kubernetes


def script_args_to_string(args):
    arg_list_string = ""
    for k, v in args.items():
        if isinstance(v, bool):
            if v:
                arg_list_string += f"\"--{k}\", "
        else:
            arg_list_string += f"\"--{k}\", "
            arg_list_string += f"\"{v}\", "
    return arg_list_string


def _handle_kubernetes_exception(copied_template_params, e):
    body = json.loads(e.body)
    if body.get("reason") == "AlreadyExists":
        print(f"{copied_template_params['job_name']} already exists!")
    else:
        print(body["message"])
        raise


def _create_job_config_from_template(template, parameters):
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    job_config_path = "_job_configs/%s" % timestamp

    pathlib.Path(job_config_path).mkdir(parents=True, exist_ok=True)

    job_config_raw = Environment(loader=FileSystemLoader("")).get_template(template).render(**parameters)

    with open("%s/%s.yaml" % (job_config_path, parameters['job_name']), 'w') as job_config_file:
        job_config_file.write(job_config_raw)

    job_config = yaml.load(job_config_raw, Loader=yaml.Loader)

    return job_config


class KubernetesHelper:
    def __init__(self):
        self.client = kubernetes.client
        self.namespace = "studachter"
        kubernetes.config.load_config()
        self.batch_v1 = kubernetes.client.BatchV1Api(kubernetes.client.ApiClient())
        self.core_v1 = kubernetes.client.CoreV1Api()

    def create_jobs(self, num_jobs: int, template, template_parameters):
        print("start spawn jobs")

        for job_idx in range(num_jobs):
            copied_template_params = template_parameters.copy()
            copied_template_params['job_name'] = f"{copied_template_params['job_name']}-{job_idx}"

            job_config = _create_job_config_from_template(template, copied_template_params)

            try:
                resp = self.batch_v1.create_namespaced_job(body=job_config, namespace=self.namespace)
                job_name = resp.metadata.name
                print(f"Created job <{job_name}>")
            except kubernetes.client.exceptions.ApiException as e:
                _handle_kubernetes_exception(copied_template_params, e)

    def save_logs(self, pod_name_matcher=None):
        all_pods = self.core_v1.list_namespaced_pod(self.namespace).items
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        save_logs_path = f"_saved_logs/{timestamp}"
        for pod in all_pods:
            pod_name = pod.metadata.name
            if pod_name_matcher is not None and pod_name_matcher in pod_name:
                try:
                    pod_status = pod.status.phase
                    pod_logs = self.core_v1.read_namespaced_pod_log(pod_name, self.namespace)
                    pathlib.Path(save_logs_path).mkdir(parents=True, exist_ok=True)
                    save_logs_file = f"{save_logs_path}/{pod_name}-{pod_status}.txt"
                    with open(save_logs_file, 'w') as f:
                        f.write(pod_logs)
                    print(f"saved logs to <{save_logs_file}>")
                except Exception as e:
                    logging.exception(e, exc_info=True)

    def delete_jobs(self, job_name_matcher=None, ignore_actives=True):
        all_jobs = self.batch_v1.list_namespaced_job(self.namespace).items
        all_pods = self.core_v1.list_namespaced_pod(self.namespace).items
        for job in all_jobs:
            job_name = job.metadata.name
            active = job.status.active
            if (job_name_matcher is None or job_name_matcher in job_name) \
                    and not (ignore_actives and active is not None):
                self.batch_v1.delete_namespaced_job(job_name, self.namespace)
                print(f"DELETE {job_name}")
                pods_to_delete = [pod.metadata.name for pod in all_pods
                                  if pod.metadata.labels['job-name'] == job_name]
                for pod_to_delete in pods_to_delete:
                    self.core_v1.delete_namespaced_pod(pod_to_delete, self.namespace)
                print(f"DELETED {pods_to_delete}")

    def delete_pods(self, pod_name_matcher=None, ignore_actives=True):
        all_pods = self.core_v1.list_namespaced_pod(self.namespace).items
        for pod in all_pods:
            pod_name = pod.metadata.name
            status = pod.status.phase
            if (pod_name_matcher is None or pod_name_matcher in pod_name) \
                    and not (ignore_actives and status != "Failed"):
                self.core_v1.delete_namespaced_pod(pod_name, self.namespace)
                print(f"DELETED {pod_name}")




