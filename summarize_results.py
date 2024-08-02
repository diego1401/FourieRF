import os

def parse_mean_file(file_path):
    """Parse mean.txt file and return the metrics."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
        if len(lines) == 4:
            psnr = float(lines[0].strip())
            ssim = float(lines[1].strip())
            l_a = float(lines[2].strip()) #NOT USED
            l_v = float(lines[3].strip())
            return psnr, ssim, l_v
    return None

def collect_metrics(root_dir):
    """Walk through directories to find mean.txt files and collect metrics."""
    metrics = {'3': {}, '6': {}, '9': {}}
    for root, dirs, files in os.walk(root_dir):
        if 'mean.txt' in files:
            mean_file_path = os.path.join(root, 'mean.txt')
            metrics_data = parse_mean_file(mean_file_path)
            if metrics_data:
                for key in metrics.keys():
                    if f'/{key}/' in root:
                        scene_name = root.split('/')[-2]
                        metrics[key][scene_name] = metrics_data
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
            # l_a_sum += metric[2]
            l_v_sum += metric[2]
        n = len(metrics[key])
        means[key] = (psnr_sum / n, ssim_sum / n,l_v_sum / n)
    return means

def write_latex_lines(metrics, means):
    """Write the metrics and means into LaTeX table lines."""
    methods = {}
    for view, scenes in metrics.items():
        for scene, metric in scenes.items():
            if scene not in methods:
                methods[scene] = {}
            methods[scene][view] = metric
    
    latex_lines = []
    for method, views in methods.items():
        line = f"{method} &"
        for metric_index in range(3):  # Iterate over PSNR, SSIM, l_a, l_v
            for view in ['3', '6', '9']:
                if view in views:
                    metric = views[view][metric_index]
                    line += f" {metric:.3f} &"
                else:
                    line += " - &"
        latex_lines.append(line.strip('&') + " \\\\")
    
    latex_lines.append("\\hline")
    mean_line = "Mean &"
    for metric_index in range(3):  # Iterate over PSNR, SSIM, l_a, l_v
        for view in ['3', '6', '9']:
            if view in means:
                metric = means[view][metric_index]
                mean_line += f" {metric:.3f} &"
            else:
                mean_line += " - &"
    latex_lines.append(mean_line.strip('&') + " \\\\")
    
    return latex_lines

root_directory = 'smooth_trajectory_ablation/increase_until_10000'  # Replace with your root directory
metrics = collect_metrics(root_directory)
means = compute_mean(metrics)
latex_lines = write_latex_lines(metrics,means)

for line in latex_lines:
    print(line)
