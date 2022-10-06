from turtle import forward
import torch 
import scipy.interpolate as si

class Bspline:
    def __init__(self,
                 x, 
                 df,#n_bases
                 degree=3,#polynomial degree
                 lower_bound =None,
                 upper_bound = None,
                 knots = None,
                 intercept = False):
        """_summary_

        Args:
            x (1-D tensor): _description_
            df (int): _description_
            lower_bound (float,optional):
            upper_bound (float,optional): _description_. Defaults to None.
            knots (1-D tensor, optional): _description_. Defaults to None.
            intercept (bool):_..._. Defaulta to None
        """
        
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.x = x
        self.df = df
        
        self.degree = degree
        
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
        if (lower_bound==None) & (upper_bound==None):#if start and end are not given,
            self.lower_bound = x.min().item()
            self.upper_bound = x.max().item()
        
        self.knots = knots
        
        if knots == None: #if knots are not given, then compute the knots
            x_range = self.upper_bound-self.lower_bound
            down = self.lower_bound - x_range*0.001
            up = self.upper_bound + x_range*0.001
            
            m= self.degree -1
            nk = self.df - m
            dknots = (up - down)/(nk - 1)
            
            start_k = down - dknots * (m + 1)
            end_k = up + dknots * (m + 1)
            steps_k = nk + 2 * m + 2
            self.knots = torch.linspace(start=start_k,
                                   end = end_k,
                                   steps = steps_k)
                                   #device = self.device)
        
        self.intercept = intercept
            
    def get_knots(self):
        return self.knots
    
    def fit(self):
        tck = [self.knots, torch.zeros(self.df,
                                       #device=self.device,
                                       dtype=torch.float32), self.degree]
        X = torch.zeros([len(self.x), self.df],
                        #device=self.device, 
                        dtype=torch.float32)
        
        x = self.x.cpu().numpy()
        for i in range(self.df):
            vec = torch.zeros(self.df, dtype=torch.float32)
            vec[i] = 1.0
            tck[1] = vec
            X[:,i]=torch.from_numpy(si.splev(x,tck,der=0))#.to(device=self.device)
            
        if self.intercept==True:
            ones = torch.ones_like(X[:,:1])#,device=self.device)
            X = torch.hstack([ones,X])
            
        return X
    
def row_kron(f1:torch.tensor,
             f2:torch.tensor):
    """Outputs the row-kronecker product

    Args:
        x1 (torch.tensor): Bspline tensor for forst variable        
        x2 (torch.tensor): Bspline tensor for second variable
    Returns:
        torch.tensor: Bspline for interaction term
    """
    out = torch.tensor([])
    for i in range(f1.shape[1]):
        tmp = torch.multiply(f1[:,i].reshape(-1,1),f2)
        out = torch.hstack((out,tmp))
    
    return out


def spline_diff(X:torch.tensor,
                df:list,
                bounds:torch.tensor):
    """_summary_

    Args:
        X (torch.tensor): size N*(P*2)--> "event1","non-event1","event2","non-event2",...
        df (list): _description_
        bounds (torch.tensor): _description_

    Returns:
        _type_: _description_
    """
    tot_eff = X.shape[1]
    assert tot_eff%2 == 0, "Incorrect number of effects"
    n_eff = tot_eff/2
    assert len(df)==n_eff, "Incorrect number of degrees of freedom"
    
    rem_input = torch.zeros((len(X),sum(df)))
    
    for i in range(int(n_eff)):
        
        lb = bounds[i][0]
        ub = bounds[i][1]
        
        event_eff = Bspline(x=X[:,i*2],
                            df=df[i],
                            lower_bound=lb,
                            upper_bound=ub).fit()
        
        non_event_eff = Bspline(x=X[:,(i*2)+1],
                                df=df[i],
                                lower_bound=lb,
                                upper_bound=ub).fit()
        
        diff = event_eff-non_event_eff
        rem_input[:,df[i]*i:(df[i]*i)+df[i]]=diff
    
    return rem_input

def spline_diff_inter(X:torch.tensor,
                      df:list,
                      bounds:torch.tensor):
    """_summary_

    Args:
        X (torch.tensor): "event1","non-event1","event2,"non-event2"
        df (list): "df_event1","df_event2"
        bounds (torch.tensor): _description_
    """
    assert X.shape[1]==4, "Incorrect number of effects"
    assert len(df)==2, "Incorrect number of degrees of freedom"
    
    
    event1 = Bspline(x=X[:,0],df=df[0],lower_bound=bounds[0][0]).fit()
    event2 = Bspline(x=X[:,2],df=df[1],lower_bound=bounds[1][0]).fit()
    
    inter_ev = row_kron(f1=event1,f2=event2)
    
    non_event1 = Bspline(x=X[:,1],df=df[0],lower_bound=bounds[0][1]).fit()
    non_event2 = Bspline(x=X[:,3],df=df[1],lower_bound=bounds[1][1]).fit()
    
    inter_nonev = row_kron(f1=non_event1,f2=non_event2)
    
    out = inter_ev-inter_nonev
    
    return out

