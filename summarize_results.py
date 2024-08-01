import os

def parse_mean_file(file_path):
    """Parse mean.txt file and return the metrics."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) == 4:
            psnr = float(lines[0].strip())
            ssim = float(lines[1].strip())
            l_a = float(lines[2].strip())
            l_v = float(lines[3].strip())
            return psnr, ssim, l_a, l_v
    return None

def collect_metrics(root_dir):
    """Walk through directories to find mean.txt files and collect metrics."""
    metrics = {'3': {}, '6': {}}
    for root, dirs, files in os.walk(root_dir):
        if 'mean.txt' in files:
            mean_file_path = os.path.join(root, 'mean.txt')
            metrics_data = parse_mean_file(mean_file_path)
            if metrics_data:
                if '/3/' in root:
                    scene_name = root.split('/')[-2]
                    metrics['3'][scene_name] = metrics_data
                elif '/6/' in root:
                    scene_name = root.split('/')[-2]
                    metrics['6'][scene_name] = metrics_data
    return metrics

def compute_mean(metrics):
    """Compute the mean of the metrics."""
    means = {}
    for key in metrics:
        psnr_sum, ssim_sum, l_a_sum, l_v_sum = 0, 0, 0, 0
        for scene in metrics[key]:
            metric = metrics[key][scene]
            psnr_sum += metric[0]
            ssim_sum += metric[1]
            l_a_sum += metric[2]
            l_v_sum += metric[3]
        n = len(metrics[key])
        means[key] = (psnr_sum / n, ssim_sum / n, l_a_sum / n, l_v_sum / n)
    return means

def write_latex_table(metrics, means, output_file):
    """Write the metrics and means into a LaTeX table."""
    with open(output_file, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Scene & PSNR & SSIM & L_A & L_V \\\\\n")
        f.write("\\hline\n")
        
        # Write individual metrics
        for key in metrics:
            for scene, metric in metrics[key].items():
                f.write(f"{scene}_{key} & {metric[0]:.10e} & {metric[1]:.10e} & {metric[2]:.10e} & {metric[3]:.10e} \\\\\n")
                f.write("\\hline\n")

        # Write mean metrics
        f.write("\\hline\n")
        for key in means:
            mean = means[key]
            f.write(f"Mean {key} & {mean[0]:.10e} & {mean[1]:.10e} & {mean[2]:.10e} & {mean[3]:.10e} \\\\\n")
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\caption{Error Metrics for Directories 3 and 6}\n")
        f.write("\\end{table}\n")

root_directory = 'log_low_cap'  # Replace with your root directory
metrics = collect_metrics(root_directory)
means = compute_mean(metrics)
write_latex_table(metrics, means, f'metrics_table_{root_directory}.tex')
