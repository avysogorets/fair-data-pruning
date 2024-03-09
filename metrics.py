import json
import os


class Metrics:
    def __init__(self):
        self.metrics = {}

    def __len__(self):
        return len(self.metrics)

    def add(self, idxs, metrics):
        metrics = {
            'train': metrics['train'],
            'val': metrics['val'],
            'test': metrics['test']}
        self.metrics[len(idxs)] = {
            'metrics': metrics,
            'idxs': idxs}

    def save(self, path: str, filename: str):
        dump_path_filename = os.path.join(path, filename)
        f = open(dump_path_filename, 'w')
        json.dump(self.metrics, f)
        f.close()
    
    def load(self, path: str, filename: str):
        load_path_filename = os.path.join(path, filename)
        f = open(load_path_filename)
        metrics = json.load(f)
        f.close()
        return metrics
