import numpy as np
import sys


def progress_bar(step, total, bar_length=80):
    progress = step / total
    block = int(bar_length * progress)
    text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {int(progress * 100)}%"
    sys.stdout.write(text)
    sys.stdout.flush()
    if step == total:
        sys.stdout.write("\n")



#mcmc_sampler class
#this class implement the monte-carlo markov chain generator
class mcmc_sampler:
    def __init__(self):
        self.par_limits = []
    def load_data(self, hist, x_bin):
        self.hist = hist
        self.x_bin = x_bin
        self.first_proposition = True
    def set_sigma(self,sigma):
        self.sigma = sigma
    def make_chain(self,n_step, method):
        par_chain = []
        self.make_first_proposition()
        self.make_model()
        LL = self.compute_LL()
        for n in range(n_step):
            self.make_proposition(method)
            self.make_model()
            LL2 = self.compute_LL()
            if self.test_acceptation(LL,LL2):
                LL = LL2
                self.pars = self.test_pars
            par_chain.append(self.pars)
            if (n%100==0):
                progress_bar(n, n_step)
        progress_bar(n_step, n_step)
        return par_chain   
    def make_chain2(self,n_step, method):
        par_chain = []
        self.make_first_proposition()
        self.make_model()
        LL = self.compute_LL()
        #add burning uniform
        for n in range(10000):
            self.make_proposition("uniform")
            self.make_model()
            LL2 = self.compute_LL()
            if self.test_acceptation(LL,LL2):
                LL = LL2
                self.pars = self.test_pars
        
        for n in range(n_step):
            self.make_proposition(method)
            self.make_model()
            LL2 = self.compute_LL()
            if self.test_acceptation(LL,LL2):
                LL = LL2
                self.pars = self.test_pars
            par_chain.append(self.pars)
            if (n%100==0):
                progress_bar(n, n_step)
        progress_bar(n_step, n_step)
        return par_chain   
    def make_chain_Robbins_Monro(self,n_step, method,target_accept=0.234, alpha=0.75):
        self.make_first_proposition()
        ndim = len(self.test_pars)
        par_chain = []
        self.make_model()
        LL = self.compute_LL()
        #add burning uniform
        for n in range(10000):
            self.make_proposition("uniform")
            self.make_model()
            LL2 = self.compute_LL()
            if self.test_acceptation(LL,LL2):
                LL = LL2
                self.pars = self.test_pars

        accept_count = 0
        
        for i in range(1, n_step):
            d = np.random.randint(ndim)
            self.test_pars = self.pars.copy()
            self.test_pars[d] = np.random.normal(self.test_pars[d], self.sigma[d] )
            self.make_model()
            LL2 = self.compute_LL()
            accepted = False
            
            if self.test_acceptation(LL,LL2):
                LL = LL2
                self.pars = self.test_pars
                
                accept_count += 1
                
                accepted = True
            par_chain.append(self.pars)                

            gamma = 1 / ((i + 1) ** alpha)
            acc = 1.0 if accepted else 0.0
            self.sigma[d] = self.sigma[d]*np.exp(gamma * (acc - target_accept))
                
            if (i%100==0):
                progress_bar(i, n_step)
            
        progress_bar(n_step, n_step)
        print(f"new proposal_std {self.sigma}")
        
        acceptance_rate = accept_count / n_step
        print(f"Acceptance rate : {acceptance_rate:.2f}")
        return par_chain


        

    def make_first_proposition(self):
        if self.first_proposition == False:
            return
        p = []
        for lim in self.par_limits:
            p.append(np.random.uniform(lim[0],lim[1]))
        self.test_pars = np.array(p)
        self.pars = np.array(p)
        self.first_proposition = False
    def make_model(self):
        pass
    def compute_LL(self):
        pass
    def make_proposition(self,method):
        if( method == "normal"):
            self.make_normal_proposition()
        elif( method == "uniform"):
            self.make_uniform_proposition()
    def make_normal_proposition(self):
        self.test_pars = np.random.normal(self.pars,self.sigma)
        while((self.test_pars[:]<self.par_limits[:,0]).any()
             | (self.test_pars[:]>self.par_limits[:,1]).any()):
            self.test_pars = np.random.normal(self.pars,self.sigma)
    def make_uniform_proposition(self):
        p = []
        for lim in self.par_limits:
            p.append(np.random.uniform(lim[0],lim[1]))
        self.test_pars = np.array(p)
    def test_acceptation(self, LL, LL2 ):
        U = np.random.uniform(0.,1.,1)
        dll = np.exp(LL2-LL)
        return U < dll 

    
    
