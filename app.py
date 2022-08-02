import os.path as ops
import lightning as L
from quick_start.components import PyTorchLightningScript, ImageServeGradio
from lit_jupyter import LitJupyter

class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.lit_jupyter = LitJupyter()
        self.train_work = PyTorchLightningScript(
            script_path=ops.join(ops.dirname(__file__), "./train_script.py"),
            script_args=["--trainer.max_epochs=5"],
        )

        self.serve_work = ImageServeGradio(L.CloudCompute("cpu"))

    def run(self):
        # 1. Run the python script that trains the model
        self.train_work.run()
        self.lit_jupyter.run()

        # 2. when a checkpoint is available, deploy
        if self.train_work.best_model_path:
            self.serve_work.run(self.train_work.best_model_path)

    def configure_layout(self):
        tab_1 = {"name": "Model training", "content": self.train_work}
        tab_2 = {"name": "Interactive demo", "content": self.serve_work}
        tab_3 = {"name": "Notebook", "content": self.lit_jupyter}
        return [tab_1, tab_2, tab_3]

app = L.LightningApp(TrainDeploy())
