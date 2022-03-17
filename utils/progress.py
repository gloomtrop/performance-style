class ProgressBar:
    """
    A class for showing a progress bar.
    Is used to track the progress of for loops.
    """

    def __init__(self, total_steps, text="Status:", print_interval=20, update_all_steps=True):
        """Creates a progress bar object and prints that there has not been any progress yet."""
        self.total_steps = total_steps
        self.text = text
        if print_interval > self.total_steps:
            self.print_interval = total_steps
        else:
            self.print_interval = print_interval
        self.update_all_steps = update_all_steps
        self.show(-1)

    def show(self, current_step):
        """Prints the progress of the loop."""
        current_step += 1
        hashtags = current_step * self.print_interval // self.total_steps
        if self.update_all_steps or current_step % (self.total_steps // self.print_interval) == 0:
            print(
                f"\r{self.text} [{'#' * (hashtags)}{'.' * (self.print_interval - hashtags)}] {current_step}/{self.total_steps}",
                end="", flush=True)
        if current_step == self.total_steps:
            print(f"\r{self.text} Done" + 60 * " ")
