import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    
class CrossGraphGAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, static_dim, sie_dim):
        super(CrossGraphGAT, self).__init__()
        self.g = g
        self.fc_static = nn.Linear(static_dim, sie_dim)
        self.fc_real = nn.Linear(in_dim+sie_dim, hidden_dim)
        self.fc_update = nn.Linear(in_dim+sie_dim, hidden_dim)
        self.fc_attn = nn.Linear(hidden_dim, 1)
        
    def edge_attention(self, edges):
        e = edges.src['z2'] + edges.dst['z1']
        e = self.fc_attn(e) * edges.dst['interval']
        return {'e': F.leaky_relu(e)}
    
    def message_func(self, edges):
        return {'h_u': edges.src['h_u'], 'e': edges.data['e']}
        
    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h_u'], dim=1)
        return {'h_u': h}
        
    def forward(self, h_real, h_update, static, interval):
        self.g.ndata['h_r'] = h_real
        self.g.ndata['h_u'] = h_update
        sie = self.fc_static(static)
        z1 = self.fc_real(torch.cat((h_real, sie), dim=-1))
        z2 = self.fc_update(torch.cat((h_update, sie), dim=-1))
        self.g.ndata['z1'] = z1
        self.g.ndata['z2'] = z2
        self.g.ndata['interval'] = interval
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h_u')
    
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs))
        
class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result
    
    
class PopNet(nn.Module):
    def __init__(self, g, in_dim, static_dim, out_dim, gat_dim1, gat_dim2, satt_dim, sie_dim, tatt_dim, gru_dim, fc_dim, num_heads, device):
        super(PopNet, self).__init__()
        self.nn_r_gat1 = MultiHeadGATLayer(g, in_dim, gat_dim1, num_heads)
        self.nn_u_gat1 = MultiHeadGATLayer(g, in_dim, gat_dim1, num_heads)
        self.nn_input = nn.Linear(in_dim, in_dim)
        self.nn_satt = CrossGraphGAT(g, gat_dim1, satt_dim, static_dim, sie_dim)
        self.nn_r_gru = nn.GRUCell(gat_dim1*2+in_dim, gru_dim)
        self.nn_u_gru = nn.GRUCell(gat_dim1+in_dim, gru_dim)
        
        self.nn_u_conv1 = CausalConv1d(in_dim, 16, 3, 1, 1)
        self.nn_u_conv2 = CausalConv1d(in_dim, 16, 3, 1, 3)
        self.nn_u_conv3 = CausalConv1d(in_dim, 16, 3, 1, 5)
        
        self.nn_r_conv1 = CausalConv1d(in_dim, 16, 3, 1, 1)
        self.nn_r_conv2 = CausalConv1d(in_dim, 16, 3, 1, 3)
        self.nn_r_conv3 = CausalConv1d(in_dim, 16, 3, 1, 5)
        
        self.nn_r_tie1 = nn.Linear(in_dim, 16)
        self.nn_r_tie2 = nn.Linear(16, 48)
        
        self.nn_u_tie1 = nn.Linear(in_dim, 16)
        self.nn_u_tie2 = nn.Linear(16, 48)

        self.nn_sgate1 = nn.Linear(gat_dim1, gat_dim1)
        self.nn_sgate2 = nn.Linear(gat_dim1, gat_dim1)

        self.nn_tgate1 = nn.Linear(gru_dim, gru_dim)
        self.nn_tgate2 = nn.Linear(gru_dim, gru_dim)
        
        self.nn_r_tatt = nn.Linear(gru_dim+48, tatt_dim)
        self.nn_u_tatt = nn.Linear(gru_dim+48, tatt_dim)
        self.tatt_func = nn.Linear(tatt_dim, 1)
        
        self.nn_r_fc = nn.Linear(gru_dim*2+48, fc_dim)
        self.nn_r_output = nn.Linear(fc_dim, out_dim)
        self.nn_u_fc = nn.Linear(gru_dim+48, fc_dim)
        self.nn_u_output = nn.Linear(fc_dim, out_dim)
        
        self.nn_kl_r1 = nn.Linear(48, gru_dim)
        self.nn_kl_r2 = nn.Linear(gru_dim, gru_dim)
        self.nn_kl_u1 = nn.Linear(48, gru_dim)
        self.nn_kl_u2 = nn.Linear(gru_dim, gru_dim)
        
        self.nn_dropout = nn.Dropout(0.5)
        self.tatt_dim = tatt_dim
        self.gru_dim = gru_dim
        self.device = device
        self.loss_KL = nn.KLDivLoss(reduction='batchmean')

    def forward(self, x_r, x_u, x_static, h_r, h_u, interval=None):
        N, T, H = x_r.size()
        
        if h_r == None:
            h_r = torch.zeros(N, self.gru_dim).to(self.device)
            h_u = torch.zeros(N, self.gru_dim).to(self.device)
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_normal_(h_r, gain=gain)
            nn.init.xavier_normal_(h_u, gain=gain)            

        his_hr = []
        his_hu = []
        
        embd_r = []
        embd_u = []
        
        r_estimate = []
        u_estimate = []
        
        conv_input_r = x_r.permute(0, 2, 1)
        conv_res1_r = self.nn_r_conv1(conv_input_r)
        conv_res3_r = self.nn_r_conv2(conv_input_r)
        conv_res5_r = self.nn_r_conv3(conv_input_r)
        
        conv_input_u = x_u.permute(0, 2, 1)
        conv_res1_u = self.nn_u_conv1(conv_input_u)
        conv_res3_u = self.nn_u_conv2(conv_input_u)
        conv_res5_u = self.nn_u_conv3(conv_input_u)
        
        if interval == None:
            ft = torch.ones((N, T), dtype=torch.float32).to(self.device)
        else:
            ft = 1/(torch.log(1+torch.exp(interval)))
        
        for i in range(T):
            r_node = self.nn_r_gat1(x_r[:, i, :])
            r_node = F.elu(r_node)
            
            u_node = self.nn_u_gat1(x_u[:, i, :])
            u_node = F.elu(u_node)
            
            #S-LAtt
            agg_u_node = self.nn_satt(r_node, u_node, x_static, ft[:, i].unsqueeze(1))
            agg_u_node = F.elu(agg_u_node)
            agg_r_node = torch.cat((r_node, agg_u_node, x_r[:, i, :]), dim=-1)
            
            #Adaptive gating
            #sgate = torch.sigmoid(self.nn_sgate1(r_node) + self.nn_sgate2(agg_u_node))
            #agg_r_node = sgate * r_node + (1-sgate) * agg_u_node
            #agg_r_node = F.elu(agg_r_node)
            #agg_r_node = r_node + agg_u_node + x_r[:, i, :]
            
            u_node = torch.cat((u_node, x_u[:, i, :]), dim=-1)
            u_node = F.elu(u_node)
            
            h_r = self.nn_r_gru(agg_r_node, h_r)
            h_u = self.nn_u_gru(u_node, h_u)
            his_hu.append(h_u)
            his_hr.append(h_r)
            
            #Temporal attention
            stack_hu = torch.stack(his_hu).to(self.device).permute(1, 0, 2) # N * t * h
            
            #Learn TIE
            scale_r = self.nn_r_tie1(torch.mean(x_r[:, :i+1, :], dim=1))
            scale_r = torch.relu(scale_r)
            scale_r = self.nn_r_tie2(scale_r)
            scale_r = torch.sigmoid(scale_r)
            r_tie = torch.cat((conv_res1_r, conv_res3_r, conv_res5_r), dim=1)[:, :, :i+1]
            r_tie = r_tie * scale_r.unsqueeze(-1).expand_as(r_tie)
            r_tie = torch.mean(r_tie, dim=-1)
            
            scale_u = self.nn_u_tie1(torch.mean(x_u[:, :i+1, :], dim=1))
            scale_u = torch.relu(scale_u)
            scale_u = self.nn_u_tie2(scale_u)
            scale_u = torch.sigmoid(scale_u)
            u_tie = torch.cat((conv_res1_u, conv_res3_u, conv_res5_u), dim=1)[:, :, :i+1]
            u_tie = u_tie * scale_u.unsqueeze(-1).expand_as(u_tie)
            u_tie = torch.mean(u_tie, dim=-1)
            stack_u_tie = u_tie.unsqueeze(1).repeat(1, i+1, 1)
            stack_u_tie = stack_u_tie.reshape(N*(i+1), 48)
            
            #T-LAtt
            tatt_hu = stack_hu.reshape(N*(i+1), self.gru_dim) # N*t, h            
            tatt_hu = self.nn_u_tatt(torch.cat((tatt_hu, stack_u_tie), dim=-1)).reshape(N, i+1, self.tatt_dim)
            tatt_hr = self.nn_r_tatt(torch.cat((h_r, r_tie), dim=-1))
            tatt_hu = (tatt_hu + tatt_hr.unsqueeze(1)).reshape(N*(i+1), self.tatt_dim)
            tatt_score = self.tatt_func(tatt_hu).reshape(N, i+1) #N, t
            
            if interval == None:
                tatt_score = tatt_score * ft[:, :i+1]
            else:
                cur_ft = torch.flip(torch.arange(0, i+1, dtype=torch.float32), [0]).unsqueeze(0).repeat((N, 1))
                tatt_score = tatt_score * cur_ft[:, :i+1]
            tatt_score = F.softmax(tatt_score, dim=-1)
            stack_hu = torch.sum(tatt_score.unsqueeze(-1) * stack_hu, dim=1)
            
            agg_hr = torch.cat((h_r, stack_hu), dim=-1)
            
            #Adaptive gating
            #tgate = torch.sigmoid(self.nn_tgate1(h_r) + self.nn_tgate2(stack_hu))
            #agg_hr = tgate * h_r + (1-tgate) * stack_hu
            #agg_hr = h_r + stack_hu# + agg_r_node
            #h_u = h_u# + u_node
            
            embd_r.append(torch.cat((agg_hr, r_tie), dim=-1))
            embd_u.append(torch.cat((h_u, u_tie), dim=-1))            
        
        his_hr = F.softmax(torch.stack(his_hr).permute(1, 0, 2).reshape(N*T, self.gru_dim), 1)
        his_hu = F.softmax(torch.stack(his_hu).permute(1, 0, 2).reshape(N*T, self.gru_dim), 1)
        
        #Alignment module
        r_estimate = torch.cat((conv_res1_r, conv_res3_r, conv_res5_r), dim=1).permute(0, 2, 1).reshape(N*T, 48)
        r_estimate = self.nn_kl_r1(r_estimate)
        r_estimate = torch.relu(r_estimate)
        r_estimate = F.log_softmax(self.nn_kl_r2(r_estimate), 1)
        u_estimate = torch.cat((conv_res1_u, conv_res3_u, conv_res5_u), dim=1).permute(0, 2, 1).reshape(N*T, 48)
        u_estimate = self.nn_kl_u1(u_estimate)
        u_estimate = torch.relu(r_estimate)
        u_estimate = F.log_softmax(self.nn_kl_u2(u_estimate), 1)
        kl_loss = self.loss_KL(r_estimate, his_hr) + self.loss_KL(u_estimate, his_hu)
        
        
        embd_r = torch.stack(embd_r).permute(1, 0, 2)
        embd_u = torch.stack(embd_u).permute(1, 0, 2)
        output_r = embd_r.reshape(N*T, self.gru_dim*2+48)
        output_r = self.nn_dropout(output_r)
        output_r = self.nn_r_fc(output_r)
        output_r = F.elu(output_r)
        output_r = self.nn_r_output(output_r).reshape(N, T, 1)
        output_u = embd_u.reshape(N*T, self.gru_dim+48)
        output_u = self.nn_dropout(output_u)
        output_u = self.nn_u_fc(output_u)
        output_u = F.elu(output_u)
        output_u = self.nn_u_output(output_u).reshape(N, T, 1)
        
        return output_r, output_u, h_r, h_u, kl_loss