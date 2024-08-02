from .tensorBase import *
import math

class TensorVM(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVM, self).__init__(aabb, gridSize, device, **kargs)
        

    def init_svd_volume(self, res, device):
        self.plane_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
        self.line_coef = torch.nn.Parameter(
            0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
        self.basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.line_coef, 'lr': lr_init_spatialxyz}, {'params': self.plane_coef, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_features(self, xyz_sampled):

        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach()
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach()

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        return sigma_feature, app_features

    def compute_densityfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_feats = F.grid_sample(self.plane_coef[:, -self.density_n_comp:], coordinate_plane, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        line_feats = F.grid_sample(self.line_coef[:, -self.density_n_comp:], coordinate_line, align_corners=True).view(
                                        -1, *xyz_sampled.shape[:1])
        
        sigma_feature = torch.sum(plane_feats * line_feats, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)
        
        plane_feats = F.grid_sample(self.plane_coef[:, :self.app_n_comp], coordinate_plane, align_corners=True).view(3 * self.app_n_comp, -1)
        line_feats = F.grid_sample(self.line_coef[:, :self.app_n_comp], coordinate_line, align_corners=True).view(3 * self.app_n_comp, -1)
        
        
        app_features = self.basis_mat((plane_feats * line_feats).T)
        
        
        return app_features
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.plane_coef)):
            total = total + torch.mean(torch.abs(self.plane_coef[idx])) + torch.mean(torch.abs(self.line_coef[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.plane_coef)):
            total = total + reg(self.plane_coef[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = torch.tensor(0)
        return total
    
    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            # print(self.line_coef.shape, vector_comps[idx].shape)
            n_comp, n_size = vector_comps[idx].shape[:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape)
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            # print(vector_comps[idx].shape, vector_comps[idx].view(n_comp,n_size).transpose(-1,-2).shape, dotp.shape,non_diagonal.shape)
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        
        return self.vectorDiffs(self.line_coef[:,-self.density_n_comp:]) + self.vectorDiffs(self.line_coef[:,:self.app_n_comp])
    
    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.line_coef[i] = torch.nn.Parameter(
                self.line_coef[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.plane_coef[i] = torch.nn.Parameter(
                self.plane_coef[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]

            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        # plane_coef[0] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[0].data, size=(res_target[1], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[0] = torch.nn.Parameter(
        #     F.interpolate(line_coef[0].data, size=(res_target[2], 1), mode='bilinear', align_corners=True))
        # plane_coef[1] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[1].data, size=(res_target[2], res_target[0]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[1] = torch.nn.Parameter(
        #     F.interpolate(line_coef[1].data, size=(res_target[1], 1), mode='bilinear', align_corners=True))
        # plane_coef[2] = torch.nn.Parameter(
        #     F.interpolate(plane_coef[2].data, size=(res_target[2], res_target[1]), mode='bilinear',
        #                   align_corners=True))
        # line_coef[2] = torch.nn.Parameter(
        #     F.interpolate(line_coef[2].data, size=(res_target[0], 1), mode='bilinear', align_corners=True))

        return plane_coef, line_coef
    
    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.plane_coef, self.line_coef = self.up_sampling_VM(self.plane_coef, self.line_coef, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

class TensorVMSplit(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)

    def init_svd_volume(self, res, device):
        self.density_plane, self.density_line = self.init_one_svd(self.density_n_comp, self.gridSize, 0.1, device)
        self.app_plane, self.app_line = self.init_one_svd(self.app_n_comp, self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)
    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz}, {'params': self.density_plane, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz}, {'params': self.app_plane, 'lr': lr_init_spatialxyz},
                         {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def vectorDiffs(self, vector_comps):
        total = 0
        
        for idx in range(len(vector_comps)):
            n_comp, n_size = vector_comps[idx].shape[1:-1]
            
            dotp = torch.matmul(vector_comps[idx].view(n_comp,n_size), vector_comps[idx].view(n_comp,n_size).transpose(-1,-2))
            non_diagonal = dotp.view(-1)[1:].view(n_comp-1, n_comp+1)[...,:-1]
            total = total + torch.mean(torch.abs(non_diagonal))
        return total

    def vector_comp_diffs(self):
        return self.vectorDiffs(self.density_line) + self.vectorDiffs(self.app_line)
    
    def density_L1(self):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + torch.mean(torch.abs(self.density_plane[idx])) + torch.mean(torch.abs(self.density_line[idx]))# + torch.mean(torch.abs(self.app_plane[idx])) + torch.mean(torch.abs(self.density_plane[idx]))
        return total
    
    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_plane)):
            total = total + reg(self.density_plane[idx]) * 1e-2 #+ reg(self.density_line[idx]) * 1e-3
        return total
        
    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_plane)):
            total = total + reg(self.app_plane[idx]) * 1e-2 #+ reg(self.app_line[idx]) * 1e-3
        return total

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.density_plane)):
            plane_coef_point = F.grid_sample(self.density_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.density_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.app_plane)):
            plane_coef_point.append(F.grid_sample(self.app_plane[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.app_line[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef[i] = torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True))
            line_coef[i] = torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))


        return plane_coef, line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_plane, self.app_line, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_plane, self.density_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.density_plane[i] = torch.nn.Parameter(
                self.density_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )
            self.app_plane[i] = torch.nn.Parameter(
                self.app_plane[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

class TensorCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorCP, self).__init__(aabb, gridSize, device, **kargs)


    def init_svd_volume(self, res, device):
        self.density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.density_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[1], coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.density_line[2], coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)


        line_coef_point = F.grid_sample(self.app_line[0], coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[1], coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.app_line[2], coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        self.density_line, self.app_line = self.up_sampling_Vector(self.density_line, self.app_line, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)


        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.density_line[i] = torch.nn.Parameter(
                self.density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.app_line[i] = torch.nn.Parameter(
                self.app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + torch.mean(torch.abs(self.density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.density_line)):
            total = total + reg(self.density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.app_line)):
            total = total + reg(self.app_line[idx]) * 1e-3
        return total
    
class TensorFourierCP(TensorBase):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(TensorFourierCP, self).__init__(aabb, gridSize, device, **kargs)
        self.register_buffer('frequency_cap',torch.ones(len(self.vecMode)*500,dtype=int))

    def init_svd_volume(self, res, device):
        self.pre_fourier_density_line = self.init_one_svd(self.density_n_comp[0], self.gridSize, 0.2, device)
        self.pre_fourier_app_line = self.init_one_svd(self.app_n_comp[0], self.gridSize, 0.2, device)
        self.basis_mat = torch.nn.Linear(self.app_n_comp[0], self.app_dim, bias=False).to(device)


    def init_one_svd(self, n_component, gridSize, scale, device):
        line_coef = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component, gridSize[vec_id], 1))))
        return torch.nn.ParameterList(line_coef).to(device)

    
    def get_optparam_groups(self, lr_init_spatialxyz = 0.02, lr_init_network = 0.001):
        grad_vars = [{'params': self.pre_fourier_density_line, 'lr': lr_init_spatialxyz},
                     {'params': self.pre_fourier_app_line, 'lr': lr_init_spatialxyz},
                     {'params': self.basis_mat.parameters(), 'lr':lr_init_network}]
        if isinstance(self.renderModule, torch.nn.Module):
            grad_vars += [{'params':self.renderModule.parameters(), 'lr':lr_init_network}]
        return grad_vars

    # def set_frequency_cap(self,value):
    #     self.frequency_cap.fill_(value)

    def get_frequency_cap(self,idx):
        return self.frequency_cap[idx].item()
    
    @torch.no_grad()
    def increase_frequency_cap(self, max_number_of_iterations):
        '''Function to increase the frequency cap of the fourier coefficients preserved after clipping.'''
        raise ValueError("Not implemented")


    def set_frequency_cap_after_resolution_change(self,old_resolution, new_resolution):
        frequency_cap_ratio = self.frequency_cap / old_resolution
        self.frequency_cap.data  = (new_resolution * frequency_cap_ratio).int()

    def get_density_line(self, idx):
        # print(self.frequency_cap)
        f_space = torch.fft.rfft(self.pre_fourier_density_line[idx],dim=-2)
        padded_tensor = torch.zeros_like(f_space)
        f_cap = self.get_frequency_cap(idx)
        padded_tensor[:,:,:f_cap,:]  = f_space[:,:,:f_cap,:] 
        return torch.fft.irfft(padded_tensor,dim=-2)
    
    def get_app_line(self, idx):
        f_space = torch.fft.rfft(self.pre_fourier_app_line[idx],dim=-2)
        padded_tensor = torch.zeros_like(f_space)
        f_cap = self.get_frequency_cap(idx)
        padded_tensor[:,:,:f_cap,:]  = f_space[:,:,:f_cap,:] 
        return torch.fft.irfft(padded_tensor,dim=-2)
    
    def compute_densityfeature(self, xyz_sampled):

        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(self.get_density_line(0), coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.get_density_line(1), coordinate_line[[1]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.get_density_line(2), coordinate_line[[2]],
                                        align_corners=True).view(-1, *xyz_sampled.shape[:1])
        sigma_feature = torch.sum(line_coef_point, dim=0)
        
        
        return sigma_feature
    
    def compute_appfeature(self, xyz_sampled):

        coordinate_line = torch.stack(
            (xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        line_coef_point = F.grid_sample(self.get_app_line(0), coordinate_line[[0]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.get_app_line(1), coordinate_line[[1]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])
        line_coef_point = line_coef_point * F.grid_sample(self.get_app_line(2), coordinate_line[[2]],
                                                          align_corners=True).view(-1, *xyz_sampled.shape[:1])

        return self.basis_mat(line_coef_point.T)
    

    @torch.no_grad()
    def up_sampling_Vector(self, density_line_coef, app_line_coef, res_target):

        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            density_line_coef[i] = torch.nn.Parameter(
                F.interpolate(density_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))
            app_line_coef[i] = torch.nn.Parameter(
                F.interpolate(app_line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True))

        return density_line_coef, app_line_coef

    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        previous_frequency_cap = self.frequency_cap
        previous_resolution = self.pre_fourier_density_line[0].shape[-2]
        self.pre_fourier_density_line, self.pre_fourier_app_line = self.up_sampling_Vector(self.pre_fourier_density_line, 
                                                                   self.pre_fourier_app_line, 
                                                                   res_target)

        self.update_stepSize(res_target)
        self.set_frequency_cap_after_resolution_change(previous_resolution, self.pre_fourier_density_line[0].shape[-2])

        print(f'upsamping to {res_target} | frequency_cap {previous_frequency_cap} -> {self.frequency_cap}')

    

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units

        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        previous_resolution = self.pre_fourier_density_line[0].shape[-2]
        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.pre_fourier_density_line[i] = torch.nn.Parameter(
                self.pre_fourier_density_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            self.pre_fourier_app_line[i] = torch.nn.Parameter(
                self.pre_fourier_app_line[i].data[...,t_l[mode0]:b_r[mode0],:]
            )

        # update frequency cap
        # self.set_frequency_cap_after_resolution_change(previous_resolution, self.pre_fourier_density_line[0].shape[-2])

        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))

    def density_L1(self):
        total = 0
        for idx in range(len(self.pre_fourier_density_line)):
            total = total + torch.mean(torch.abs(self.pre_fourier_density_line[idx]))
        return total

    def TV_loss_density(self, reg):
        total = 0
        for idx in range(len(self.pre_fourier_density_line)):
            total = total + reg(self.pre_fourier_density_line[idx]) * 1e-3
        return total

    def TV_loss_app(self, reg):
        total = 0
        for idx in range(len(self.pre_fourier_app_line)):
            total = total + reg(self.pre_fourier_app_line[idx]) * 1e-3
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
    # raise ValueError("stop")
    # Make sure sum of values in gaussian kernel equals 1.
    # gaussian_kernel = gaussian_kernel/ torch.sum(gaussian_kernel)
    gaussian_kernel /= torch.max(gaussian_kernel)
    return gaussian_kernel

class FourierTensorVMSplit(TensorVMSplit):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(FourierTensorVMSplit, self).__init__(aabb, gridSize, device, **kargs)
        self.max_freq = 100.0
        # Fourier Frequency Handler
        self.density_clip = kargs['density_clip']
        self.color_clip = kargs['color_clip']
        assert self.density_clip > 0 and self.color_clip > 0
        assert self.density_clip <= 100.0 and self.color_clip <= 100.0
        
        self.register_buffer('frequency_cap_density',torch.tensor([self.density_clip,self.density_clip,self.density_clip]))
        self.register_buffer('frequency_cap_color',torch.tensor([self.color_clip,self.color_clip,self.color_clip]))
        sigmas =[10.0,10.0,10.0]
        
        for i,size in enumerate(gridSize):
            mat_id_0, mat_id_1 = self.matMode[i]
            self.register_buffer(f'filtering_kernel_{self.vecMode[i]}',off_center_gaussian(gridSize[mat_id_0], gridSize[mat_id_1],sigma=sigmas[i]))
    
    @torch.no_grad()
    def increase_frequency_cap(self,max_number_of_iterations):
        if (self.frequency_cap_color>=self.max_freq).all() and (self.frequency_cap_density>=self.max_freq).all(): 
            return
        delta_density = (self.max_freq - self.density_clip)/max_number_of_iterations
        delta_color = (self.max_freq - self.color_clip)/max_number_of_iterations
                                                                                                                                                                                                                                                                              
        self.fourier_cap()
        for i in range(len(self.vecMode)):
            self.density_line[i].copy_(self.density_line_capped[i])
            self.density_plane[i].copy_(self.density_planes_capped[i])
            self.app_line[i].copy_(self.app_line_capped[i])
            self.app_plane[i].copy_(self.app_planes_capped[i])

        self.frequency_cap_density = torch.clamp(self.frequency_cap_density+delta_density,0,self.max_freq)
        self.frequency_cap_color = torch.clamp(self.frequency_cap_color+delta_color,0,self.max_freq)

    @torch.no_grad()
    def visualize_fourier_space(self):
        # Cap density
        # os.makedirs(FEATURES_SPACE_PATH, exist_ok=True)
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
            
    def fourier_cap_on_lines(self,lines,frequency_cap):
        assert (frequency_cap >= 0).all()
        assert (frequency_cap <= 1).all()
        fourier_capped_lines = []
        for idx in range(len(lines)):
                # Cap lines
                fourier_line = torch.fft.rfft(lines[idx],dim=-2)
                padded_line = torch.zeros_like(fourier_line)
                
                resolution_line = fourier_line.shape[2]
                # line_frequency_cap = int(resolution_line * frequency_cap[idx])
                # line_frequency_cap = max(line_frequency_cap,1) # we do not want 0 arrays
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

    def fourier_cap_on_planes(self,planes,frequency_cap):
        assert (frequency_cap >= 0).all()
        assert (frequency_cap <= 1).all()
        fourier_capped_planes = []
        for idx in range(len(planes)):
            # Cap Planes
            fourier_plane = torch.fft.fft2(planes[idx]) #rfft2 applies the fourier transform to the last 2 dimensions
            fourier_plane_shiffted = torch.fft.fftshift(fourier_plane)
            # padded_fourier_plane = torch.zeros_like(fourier_plane_shiffted)
            _, feat_dim, _, _ = fourier_plane_shiffted.shape
            
            gaussian_blur = getattr(self,f'filtering_kernel_{self.vecMode[idx]}').clone()
            percentage = (1.0-frequency_cap[idx])
            mask = (gaussian_blur>=percentage)
            gaussian_blur[mask] = 1.0
            gaussian_blur = gaussian_blur.repeat(feat_dim,1,1).unsqueeze(0)
            # padded_fourier_plane.view(1, feat_dim,-1)[:,:,mask] = fourier_plane_shiffted.view(1, feat_dim,-1)[:,:,mask]
            
            fourier_plane_unshiffted = torch.fft.ifftshift(fourier_plane_shiffted*gaussian_blur.to(fourier_plane_shiffted.device).float())
            plane_capped = torch.real(torch.fft.ifft2(fourier_plane_unshiffted))

            fourier_capped_planes.append(plane_capped)

        return fourier_capped_planes

    def fourier_cap(self):
        '''
        This function smooths the signals encoded in the the TensoRF representation. It should be called once at the start of every iteration as
        long frequency_cap < 100.
        '''

        self.density_line_capped = self.fourier_cap_on_lines(self.density_line,self.frequency_cap_density/100.0)
        self.density_planes_capped = self.fourier_cap_on_planes(self.density_plane,self.frequency_cap_density/100.0)

        self.app_line_capped = self.fourier_cap_on_lines(self.app_line,self.frequency_cap_color/100.0)
        self.app_planes_capped = self.fourier_cap_on_planes(self.app_plane,self.frequency_cap_color/100.0)

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

    def vector_comp_diffs(self):
        raise ValueError("not implemented")
    
    def density_L1(self):
        raise ValueError("not implemented")
    
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

    @torch.no_grad()
    def up_sampling_VM(self, plane_coef, line_coef, res_target):
        new_plane = []
        new_line = []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            new_plane.append(torch.nn.Parameter(
                F.interpolate(plane_coef[i].data, size=(res_target[mat_id_1], res_target[mat_id_0]), mode='bilinear',
                              align_corners=True)))
            new_line.append(torch.nn.Parameter(
                F.interpolate(line_coef[i].data, size=(res_target[vec_id], 1), mode='bilinear', align_corners=True)))

        return torch.nn.ParameterList(new_plane), torch.nn.ParameterList(new_line)
    
    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        old_grid_size = self.gridSize
        self.app_plane, self.app_line = self.up_sampling_VM(self.app_planes_capped, self.app_line_capped, res_target)
        self.density_plane, self.density_line = self.up_sampling_VM(self.density_planes_capped, self.density_line_capped, res_target)

        self.update_stepSize(res_target)
        print(f'upsamping to {res_target}')
        new_grid_size = self.gridSize
        ratio = (new_grid_size/old_grid_size).cpu()
        print(f'upsampling ratio {ratio}')
        self.frequency_cap_color *= ratio
        self.frequency_cap_density *= ratio
        self.frequency_cap_density = torch.clamp(self.frequency_cap_density,0,self.max_freq)
        self.frequency_cap_color = torch.clamp(self.frequency_cap_color,0,self.max_freq)
        # update filters
        for i,size in enumerate(res_target):
            mat_id_0, mat_id_1 = self.matMode[i]
            setattr(self,f'filtering_kernel_{self.vecMode[i]}',off_center_gaussian(res_target[mat_id_0], res_target[mat_id_1]))

class FourierTensorVM(TensorVM):
    def __init__(self, aabb, gridSize, device, **kargs):
        super(FourierTensorVM, self).__init__(aabb, gridSize, device, **kargs)
        # Fourier Frequency Handler
        # cap = 5.0 # lego cap
        cap = 5.0
        self.register_buffer('frequency_cap',torch.tensor([cap,100.0,100.0]))
        for i,size in enumerate(gridSize):
            mat_id_0, mat_id_1 = self.matMode[i]
            self.register_buffer(f'filtering_kernel_{self.vecMode[i]}',off_center_gaussian(gridSize[mat_id_1], gridSize[mat_id_0]))

    def init_svd_volume(self, res, device):
        self.plane_coef, self.line_coef = self.init_one_svd(tuple(self.app_n_comp)+tuple(self.density_n_comp), self.gridSize, 0.1, device)
        self.basis_mat = torch.nn.Linear(sum(self.app_n_comp), self.app_dim, bias=False).to(device)
        
    def increase_frequency_cap(self,max_number_of_iterations):
        raise ValueError("Not implemented")

    def init_one_svd(self, n_component, gridSize, scale, device):
        plane_coef, line_coef = [], []
        for i in range(len(self.vecMode)):
            vec_id = self.vecMode[i]
            mat_id_0, mat_id_1 = self.matMode[i]
            plane_coef.append(torch.nn.Parameter(
                scale * torch.randn((1, n_component[i], gridSize[mat_id_1], gridSize[mat_id_0]))))  #
            line_coef.append(
                torch.nn.Parameter(scale * torch.randn((1, n_component[i], gridSize[vec_id], 1))))

        return torch.nn.ParameterList(plane_coef).to(device), torch.nn.ParameterList(line_coef).to(device)

    def fourier_cap(self):
        '''
        This function smooths the signals encoded in the the TensoRF representation. It should be called once at the start of every iteration as
        long frequency_cap < 100.
        '''
        # Cap density
        new_lines = []
        new_planes = []
        for idx in range(len(self.plane_coef)):
            f_space = torch.fft.rfft(self.line_coef[idx],dim=-2)
            
            padded_tensor_line = torch.zeros_like(f_space)
            
            f_cap = int(f_space.shape[2] * self.frequency_cap[idx]/100.0)
            padded_tensor_line[:,:,:f_cap,:]  = f_space[:,:,:f_cap,:] 
            new_line_coef = torch.fft.irfft(padded_tensor_line,dim=-2)
            # Cap Planes
            f_space = torch.fft.fft2(self.plane_coef[idx]) #rfft2 applies the fourier transform to the last 2 dimensions
            f_space_shifted = torch.fft.fftshift(f_space)
            
            padded_tensor_plane = torch.zeros_like(f_space_shifted)
            
            _, feat_dim, _, _ = f_space_shifted.shape
            mask = (getattr(self,f'filtering_kernel_{self.vecMode[idx]}')>=(1-self.frequency_cap[idx]/100.0)).flatten()
            padded_tensor_plane.view(1, feat_dim,-1)[:,:,mask] = f_space_shifted.view(1, feat_dim,-1)[:,:,mask]

            padded_tensorf_unshifted = torch.fft.ifftshift(padded_tensor_plane)
            new_plane = torch.real(torch.fft.ifft2(padded_tensorf_unshifted))
            # Store them
            new_lines.append(new_line_coef)
            new_planes.append(new_plane)

        self.line_capped = new_lines
        self.planes_capped = new_planes

    def compute_densityfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        sigma_feature = torch.zeros((xyz_sampled.shape[0],), device=xyz_sampled.device)
        for idx_plane in range(len(self.planes_capped)):
            plane_coef_point = F.grid_sample(self.planes_capped[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1])
            line_coef_point = F.grid_sample(self.line_capped[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1])
            sigma_feature = sigma_feature + torch.sum(plane_coef_point * line_coef_point, dim=0)

        return sigma_feature

    def compute_appfeature(self, xyz_sampled):

        # plane + line basis
        coordinate_plane = torch.stack((xyz_sampled[..., self.matMode[0]], xyz_sampled[..., self.matMode[1]], xyz_sampled[..., self.matMode[2]])).detach().view(3, -1, 1, 2)
        coordinate_line = torch.stack((xyz_sampled[..., self.vecMode[0]], xyz_sampled[..., self.vecMode[1]], xyz_sampled[..., self.vecMode[2]]))
        coordinate_line = torch.stack((torch.zeros_like(coordinate_line), coordinate_line), dim=-1).detach().view(3, -1, 1, 2)

        plane_coef_point,line_coef_point = [],[]
        for idx_plane in range(len(self.planes_capped)):
            plane_coef_point.append(F.grid_sample(self.planes_capped[idx_plane], coordinate_plane[[idx_plane]],
                                                align_corners=True).view(-1, *xyz_sampled.shape[:1]))
            line_coef_point.append(F.grid_sample(self.line_capped[idx_plane], coordinate_line[[idx_plane]],
                                            align_corners=True).view(-1, *xyz_sampled.shape[:1]))
        plane_coef_point, line_coef_point = torch.cat(plane_coef_point), torch.cat(line_coef_point)


        return self.basis_mat((plane_coef_point * line_coef_point).T)


    @torch.no_grad()
    def upsample_volume_grid(self, res_target):
        super(FourierTensorVM, self).upsample_volume_grid(res_target)
        # update filters
        for i,size in enumerate(res_target):
            mat_id_0, mat_id_1 = self.matMode[i]
            setattr(self,f'filtering_kernel_{self.vecMode[i]}',off_center_gaussian(res_target[mat_id_1], res_target[mat_id_0]))

    @torch.no_grad()
    def shrink(self, new_aabb):
        print("====> shrinking ...")
        xyz_min, xyz_max = new_aabb
        t_l, b_r = (xyz_min - self.aabb[0]) / self.units, (xyz_max - self.aabb[0]) / self.units
        # print(new_aabb, self.aabb)
        # print(t_l, b_r,self.alphaMask.alpha_volume.shape)
        t_l, b_r = torch.round(torch.round(t_l)).long(), torch.round(b_r).long() + 1
        b_r = torch.stack([b_r, self.gridSize]).amin(0)

        for i in range(len(self.vecMode)):
            mode0 = self.vecMode[i]
            self.line_coef[i] = torch.nn.Parameter(
                self.line_coef[i].data[...,t_l[mode0]:b_r[mode0],:]
            )
            mode0, mode1 = self.matMode[i]
            self.plane_coef[i] = torch.nn.Parameter(
                self.plane_coef[i].data[...,t_l[mode1]:b_r[mode1],t_l[mode0]:b_r[mode0]]
            )


        if not torch.all(self.alphaMask.gridSize == self.gridSize):
            t_l_r, b_r_r = t_l / (self.gridSize-1), (b_r-1) / (self.gridSize-1)
            correct_aabb = torch.zeros_like(new_aabb)
            correct_aabb[0] = (1-t_l_r)*self.aabb[0] + t_l_r*self.aabb[1]
            correct_aabb[1] = (1-b_r_r)*self.aabb[0] + b_r_r*self.aabb[1]
            print("aabb", new_aabb, "\ncorrect aabb", correct_aabb)
            new_aabb = correct_aabb

        newSize = b_r - t_l
        self.aabb = new_aabb
        self.update_stepSize((newSize[0], newSize[1], newSize[2]))
        newSize = newSize.cpu()
        # update filters
        for i,size in enumerate(newSize):
            mat_id_0, mat_id_1 = self.matMode[i]
            setattr(self,f'filtering_kernel_{self.vecMode[i]}',off_center_gaussian(newSize[mat_id_1], newSize[mat_id_0]))