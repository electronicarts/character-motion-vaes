import csv


class CSVLogger(object):
    def __init__(self, log_path):
        self.csvfile = open(log_path, "w")
        self.writer = None

    def init_writer(self, keys):
        if self.writer is None:
            self.writer = csv.DictWriter(
                self.csvfile, fieldnames=list(keys), lineterminator="\n"
            )
            self.writer.writeheader()

    def log_epoch(self, data):
        self.init_writer(data.keys())
        self.writer.writerow(data)
        self.csvfile.flush()

    def __del__(self):
        self.csvfile.close()


class ConsoleCSVLogger(CSVLogger):
    def __init__(self, console_log_interval=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console_log_interval = console_log_interval

    def log_epoch(self, data):
        super().log_epoch(data)

        if data["iter"] % self.console_log_interval == 0:
            print(
                (
                    "Updates {}, num timesteps {}, "
                    "FPS {}, mean/median reward {:.1f}/{:.1f}, "
                    "min/max reward {:.1f}/{:.1f}, "
                    "policy loss {:.5f}"
                ).format(
                    data["iter"],
                    data["total_num_steps"],
                    data["fps"],
                    data["mean_rew"],
                    data["median_rew"],
                    data["min_rew"],
                    data["max_rew"],
                    data["action_loss"],
                ),
                flush=True,
            )
