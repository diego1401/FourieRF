from .tensorBase import *
from .tensoRF import TensorVMSplit, TensorCP
import math

    
# FourieRF ################################

def fourier_on_lines(lines,frequency_cap):
        assert (frequency_cap >= 0).all()
        assert (frequency_cap <= 1).all()
        fourier_capped_lines = []
        for idx in range(len(lines)):
            # Cap lines
            fourier_line = torch.fft.rfft(lines[idx],dim=-2)
            padded_line = torch.zeros_like(fourier_line)
            
            resolution_line = fourier_line.shape[2]
            line_frequency_cap_continous = resolution_line * frequency_cap[idx]
            if line_frequency_cap_continous < 1:
                line_frequency_cap = 1
                padded_line[:,:,:line_frequency_cap,:] = fourier_line[:,:,:line_frequency_cap,:]
            elif line_frequency_cap_continous >= resolution_line:
                padded_line[...] = fourier_line[...]
            else:
                line_frequency_cap = int(line_frequency_cap_continous)
                reminder_frequency = line_frequency_cap_continous - line_frequency_cap
                padded_line[:,:,:line_frequency_cap,:] = fourier_line[:,:,:line_frequency_cap,:]
                # adding remainder
                padded_line[:,:,line_frequency_cap,:] = reminder_frequency * fourier_line[:,:,line_frequency_cap,:]
                
            line_capped = torch.fft.irfft(padded_line,dim=-2,n=lines[idx].shape[-2])
            # Store them
            fourier_capped_lines.append(line_capped)
                
        return fourier_capped_lines
    
class TensorFourierCP(TensorCP):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorFourierCP, self).__init__(aabb, gridSize, device, **kargs)
        self.max_freq = 1.0
        # Fourier Frequency Handler
        self.density_clip = kargs['density_clip']
        self.color_clip = kargs['color_clip']
        assert self.density_clip > 0 and self.color_clip > 0
        assert self.density_clip <= 100.0 and self.color_clip <= 100.0

        self.density_clip /= 100.0
        self.color_clip /= 100.0
        
        self.register_buffer('frequency_cap_density',torch.tensor([self.density_clip,self.density_clip,self.density_clip]))
        self.register_buffer('frequency_cap_color',torch.tensor([self.color_clip,self.color_clip,self.color_clip]))
        
    def visualize_fourier_space(self):
        return
        
    def percentage_of_parameters(self):
        return self.frequency_cap_density[0]
    
    @torch.no_grad()
    def increase_frequency_cap(self,max_number_of_iterations):
        if (self.frequency_cap_color>=self.max_freq).all() and (self.frequency_cap_density>=self.max_freq).all(): 
            return
        delta_density = (self.max_freq - self.density_clip)/max_number_of_iterations
        delta_color = (self.max_freq - self.color_clip)/max_number_of_iterations
        
        self.frequency_cap_density = torch.clamp(self.frequency_cap_density+delta_density,0,self.max_freq)
        self.frequency_cap_color = torch.clamp(self.frequency_cap_color+delta_color,0,self.max_freq)
        
    def fourier_cap(self):
        '''
        This function smooths the signals encoded in the the TensoRF representation. It should be called once at the start of every iteration as
        long frequency_cap < 100.
        '''

        self.density_line_capped = fourier_on_lines(self.density_line,self.frequency_cap_density)
        self.app_line_capped = fourier_on_lines(self.app_line,self.frequency_cap_color)
    
    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(self.density_line_capped[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line_capped[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line_capped[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(self.app_line_capped[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line_capped[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line_capped[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line_capped)):
            total = total + torch.mean(torch.abs(self.density_line_capped[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line_capped)):
            total = total + reg(self.density_line_capped[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line_capped)):
            total = total + reg(self.app_line_capped[idx]) * 1e-3
        return total

def off_center_gaussian(size1,size2,sigma=40.0):
    '''
    Return off centered gaussian to much the DC of the 2D fft. The kernel returned is normalized as to obtain values between [0,1]
    '''
    kernel_size = np.array([size1,size2])
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size[0])
    x_grid = x_cord.repeat(kernel_size[1]).view(kernel_size[1],kernel_size[0])
    y_cord = torch.arange(kernel_size[1])
    y_grid = y_cord.repeat(kernel_size[0]).view(kernel_size[0], kernel_size[1]).t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    # gaussian_kernel = gaussian_kernel/ torch.sum(gaussian_kernel)
    # gaussian_kernel /= torch.max(gaussian_kernel)
    return gaussian_kernel.float()

def create_circle_array(n, m, r_ratio):
    """
    Create an nxm array with a circle of radius r marked with 1.0 inside and 0.0 outside.
    
    Parameters:
    n (int): Number of rows.
    m (int): Number of columns.
    r_ratio (float): Ratio of the radius of the circle to the smaller dimension of the array.
    
    Returns:
    numpy.ndarray: The resulting array.
    """
    # Calculate the radius
    square_side=max(n, m)
    diag = np.sqrt(2*square_side**2)
    # radius = np.sqrt(r_ratio/2 * diag)
    radius=r_ratio/2 * diag
    # Create an nxm array of zeros
    if r_ratio >= 1.0:
        array = np.ones((n, m))
        return torch.from_numpy(array).float() + 1e-6
    array = np.zeros((n, m))
    
    center_x, center_y = m// 2,n // 2
    
    # Create a coordinate grid
    y, x = np.ogrid[:n, :m]
    
    # Calculate the distance of each point from the center
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    
    # Set the values within the radius to 1.0
    array[distance_from_center <= radius] = 1.0
    
    return torch.from_numpy(array).float() + 1e-6

def fourier_on_planes(planes,filters:list):
        fourier_capped_planes = []
        for idx in range(len(planes)):
            # Cap Planes
            fourier_plane = torch.fft.fft2(planes[idx]) #rfft2 applies the fourier transform to the last 2 dimensions
            fourier_plane_shiffted = torch.fft.fftshift(fourier_plane)
            _, feat_dim, _, _ = fourier_plane_shiffted.shape
            
            filter_kernel = filters[idx].repeat(feat_dim,1,1).unsqueeze(0)
            fourier_plane_unshiffted = torch.fft.ifftshift(fourier_plane_shiffted*filter_kernel.to(fourier_plane_shiffted.device))
            plane_capped = torch.real(torch.fft.ifft2(fourier_plane_unshiffted))

            fourier_capped_planes.append(plane_capped)

        return fourier_capped_planes

class FourierTensorVMSplit(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(FourierTensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        self.max_freq = 1.0
        # Fourier Frequency Handler
        self.density_clip = kargs['density_clip']
        self.color_clip = kargs['color_clip']
        assert self.density_clip > 0 and self.color_clip > 0
        assert self.density_clip <= 100.0 and self.color_clip <= 100.0

        self.density_clip /= 100.0
        self.color_clip /= 100.0
        
        self.register_buffer('frequency_cap_density',torch.tensor([self.density_clip,self.density_clip,self.density_clip]))
        self.register_buffer('frequency_cap_color',torch.tensor([self.color_clip,self.color_clip,self.color_clip]))

        

        for i,size in enumerate(gridSize):
            mat_id_0, mat_id_1 = self.matMode[i]
            gaussian_blur = create_circle_array(gridSize[mat_id_1], gridSize[mat_id_0],r_ratio=1.0)
            self.register_buffer(f'filtering_kernel_{self.vecMode[i]}',gaussian_blur)
        
    @torch.no_grad()
    def increase_frequency_cap(self,max_number_of_iterations):
        if (self.frequency_cap_color>=self.max_freq).all() and (self.frequency_cap_density>=self.max_freq).all(): 
            return
        delta_density = (self.max_freq - self.density_clip)/max_number_of_iterations
        delta_color = (self.max_freq - self.color_clip)/max_number_of_iterations

        self.frequency_cap_density = torch.clamp(self.frequency_cap_density+delta_density,0,self.max_freq)
        self.frequency_cap_color = torch.clamp(self.frequency_cap_color+delta_color,0,self.max_freq)
        
        self.update_filters()

    @torch.no_grad()
    def visualize_fourier_space(self):
        # Cap density
        lines, f_lines, planes, f_planes = [],[],[],[]
        for idx in range(len(self.density_plane)):
            # Lines
            density_line = self.density_line[idx][0].mean(0)
            density_line /= density_line.max()
            density_line = density_line.cpu()

            app_line = self.app_line[idx][0].mean(0)
            app_line /= app_line.max()
            app_line = app_line.cpu()

            line = np.concatenate([density_line,app_line],axis=0)
            # Fourier lines
            density_f_space_line = torch.fft.rfft(self.density_line[idx],dim=-2)[0,...,0]
            density_f_space_line = torch.abs(density_f_space_line)
            density_f_space_line /= density_f_space_line.max()
            density_f_space_line = density_f_space_line.cpu()

            app_f_space_line = torch.fft.rfft(self.app_line[idx],dim=-2)[0,...,0]
            app_f_space_line = torch.abs(app_f_space_line)
            app_f_space_line /= app_f_space_line.max()
            app_f_space_line = app_f_space_line.cpu()

            f_line = np.concatenate([density_f_space_line,app_f_space_line],axis=0)
            # Planes
            density_plane = self.density_plane[idx][0].mean(0)
            density_plane /= density_plane.max()
            density_plane = density_plane.cpu().numpy().astype(np.float32)

            app_plane = self.app_plane[idx][0].mean(0)
            app_plane /= app_plane.max()
            app_plane = app_plane.cpu().numpy().astype(np.float32)

            plane = np.concatenate([density_plane,app_plane],axis=1)
            # Fourier Planes
            density_f_space = torch.fft.fft2(self.density_plane[idx]) #rfft2 applies the fourier transform to the last 2 dimensions
            density_f_space = torch.abs(torch.fft.fftshift(density_f_space)[0].mean(0))#.permute(1,2,0)
            density_f_space /= density_f_space.max()
            density_f_space = density_f_space.cpu().numpy().astype(np.float32)

            app_f_space = torch.fft.fft2(self.app_plane[idx]) #rfft2 applies the fourier transform to the last 2 dimensions
            app_f_space = torch.abs(torch.fft.fftshift(app_f_space)[0].mean(0))#.permute(1,2,0)
            app_f_space /= app_f_space.max()
            app_f_space = app_f_space.cpu().numpy().astype(np.float32)

            f_plane = np.concatenate([density_f_space,app_f_space],axis=1)

            lines.append(line)
            f_lines.append(f_line)
            planes.append(plane)
            f_planes.append(f_plane)

        return lines,f_lines,planes,f_planes

    def percentage_of_parameters(self):
        gaussian_blur = getattr(self,f'filtering_kernel_{self.vecMode[0]}').clone()
        mask = (gaussian_blur>=0.5)
        return mask.sum()/(gaussian_blur.shape[0]*gaussian_blur.shape[1])
    
    def fourier_cap(self):
        '''
        This function smooths the signals encoded in the the TensoRF representation. It should be called once at the start of every iteration as
        long frequency_cap < 100.
        '''

        filter_kernel_list = [getattr(self,f'filtering_kernel_{self.vecMode[idx]}') for idx in range(len(self.vecMode))]
        self.density_line_capped = fourier_on_lines(self.density_line,self.frequency_cap_density)
        self.density_planes_capped = fourier_on_planes(self.density_plane,filter_kernel_list)

        self.app_line_capped = fourier_on_lines(self.app_line,self.frequency_cap_color)
        self.app_planes_capped = fourier_on_planes(self.app_plane,filter_kernel_list)

    def compute_densityfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_planes_capped)):
            plane_coef_point = F.grid_sample(self.density_planes_capped[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line_capped[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):
        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_planes_capped)):
            plane_coef_point.append(F.grid_sample(self.app_planes_capped[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line_capped[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_planes_capped)):
            total = total + torch.mean(torch.abs(self.density_planes_capped[idx])) + torch.mean(torch.abs(self.density_line_capped[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_planes_capped)):
            total = total + reg(self.density_planes_capped[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_planes_capped)):
            total = total + reg(self.app_planes_capped[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    def update_filters(self):
        for i,size in enumerate(self.gridSize):
            mat_id_0, mat_id_1 = self.matMode[i]
            
            filter_kernel = create_circle_array(self.gridSize[mat_id_1].item(), self.gridSize[mat_id_0].item(),
                                               r_ratio=self.frequency_cap_density[i].item()).to(self.gridSize.device)
            setattr(self,f'filtering_kernel_{self.vecMode[i]}',filter_kernel)
