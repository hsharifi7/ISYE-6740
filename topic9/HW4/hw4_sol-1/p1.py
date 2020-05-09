"""
This is a sample code for ISYE 6740-OAN Homework4 Problem 1 Gaussian Mixture model

Data:
- Description: Number "6", "2" from MINST dataset
- Location: ./data/data.dat, ./data/label.dat

Requirements:
- Python: 3.7.3
- arrow: 0.13.1
- numpy: 1.16.5
- matplotlib: 3.0.3

Author: Ruyi Ding
"""
import arrow
import numpy as np 
import matplotlib.pyplot as plt

class GaussianMixtureModel(object):
    def __init__(self, data, label, C, iternum=100, r=50):
        """
        initialize the input data, labels
        """
        self.data      = data
        self.data_dims = data.shape[0]
        self.data_num  = data.shape[1]
        self.label     = label
        self.C         = C
        self.iternum   = iternum
        self.r         = r
        self.logliks   = []

    def _initialization(self):
        """
        initialization for the parameters of GMM model
        """
        # initialize pi
        self.pis   = np.random.rand(1, 2)
        
        # initialize mean
        self.mus   = np.random.normal(
            size=[self.data_dims, self.C])             # [data_dims, C]
        
        # initialize covariance matrix
        covs = []
        for k in range(self.C):
            cov   = np.eye(self.data_dims)             # [data_dims, data_dims]
            covs.append(cov)
        self.covs = np.stack(covs, axis=-1)            # [data_dims, data_dims, C]
    
    def _speedup_density(self, k):
        """
        Speed up probability density function given k class 
        """

        x     = self.data                              # [data_dims, data_num]            
        
        # mean of class k and tiled to the same shape as data                 
        mu_k  = np.tile(np.expand_dims(self.mus[:, k], axis=1), 
            reps=[1, self.data_num])                   # [data_dims, data_num]
        
        # covariance matrix of class k
        cov_k = self.covs[:, :, k]                     # [data_dims, data_dims]

        # eigen-decomposition
        eigenValues, eigenVectors = np.linalg.eigh(cov_k)

        idx = eigenValues.argsort()[::-1]   
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]

        # low rank estimator of covariance matrix, the number of dimension used is controlled by self.r
        est_covk = eigenVectors[:, :self.r] @ np.diag(eigenValues[:self.r]) @ eigenVectors[:, :self.r].T

        # np.power((2 * np.pi), self.data_dims) will be cancelled out when calculate self.tau
        term1    = 1/np.sqrt(eigenValues[:self.r].prod())
        
        est_x    = eigenVectors[:, :self.r].T @ x
        est_muk  = eigenVectors[:, :self.r].T @ mu_k

        term2    = np.exp(-1/2*np.sum(np.square(est_x-est_muk)\
        /np.tile(np.expand_dims(eigenValues[:self.r], axis=1), 
            reps=[1, self.data_num]), axis=0))                 # [data_num, ]

        return term1 * term2                                   # [data_num, ]   

    def Estep(self):
        """
        E-step
        """
        lik_ks = []
        for k in range(self.C):
            lik_k = self._speedup_density(k)             # [data_num,] 
            pi_k  = self.pis[0, k]                           
            lik_ks.append(pi_k * lik_k)
        # likelihood
        lik_ks  = np.stack(lik_ks, axis=1).T             # [C, data_num]
        taus = lik_ks/np.tile(np.sum(lik_ks, axis=0), reps=[self.C, 1])

        # loglikelihood, with the constant part
        logliks = np.log(lik_ks) - self.da    # Baseline model KMeans
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=1).fit(data.T)
    idxs   = kmeans.labels_

    preds = []
    for idx in idxs:
        if idx == 1:
            preds.append(6.)
        else:
            preds.append(2.)
    preds = np.asarray(preds)

    correct2 = 0
    wrong2   = 0
    correct6 = 0
    wrong6   = 0
    for idx, label in enumerate(label):
        if label == 2.:
            if preds[idx] == label:
                correct2 += 1
            else:
                wrong2 += 1
        elif label == 6.:
            if preds[idx] == label:
                correct6 += 1
            else:
                wrong6 += 1
    mcr2 = wrong2/(wrong2+correct2)
    mcr6 = wrong6/(wrong6+correct6)
    print(f"Kmeans: label2 : correct:{correct2} wrong:{wrong2} mcr:{mcr2}")
    print(f"Kmeans: label6 : correct:{correct6} wrong:{wrong6} mcr:{mcr6}")  ta_dims/2 * np.log(2*np.pi)        
        self.logliks.append(np.sum(taus * logliks))

        return taus
    
    def Mstep(self):
        """
        M-step
        """
        taus = self.taus                              # [C, data_num]
        
        # Update Pic
        pis  = np.zeros(shape=[1, self.C])
        for k in range(self.C):
            pis[:, k] = taus[k, :].sum()/self.data_num
        self.pis = pis                                # [1, C]
        
        # Update mean
        mus  = np.zeros(shape=[self.data_dims, self.C])
        for k in range(self.C):
            tau_k = taus[k, :].reshape(1, self.data_num)      # [1, data_num]
            tau_k = np.tile(tau_k, reps=[self.data_dims, 1])  # [data_dim, data_num]  
            x     = self.data                                 # [data_dim, data_num]
            nominator   = np.sum(x * tau_k, axis=1)           # [data_dim, ]
            denominator = taus[k, :].sum() 
            mus[:, k]   = nominator/denominator
        self.mus = mus

        # Update covariance
        covs = np.zeros(shape=[self.data_dims, self.data_dims, self.C])
        for k in range(self.C):
            tau_k = taus[k, :].reshape(1, self.data_num)       # [1, data_num]
            tau_k = np.tile(tau_k, reps=[self.data_dims, 1])   # [data_dim, data_num]
            mu_k  = self.mus[:, k].reshape(-1, 1)              # [data_dim, 1]
            mu_k  = np.tile(mu_k, reps=[1, self.data_num])     # [data_dim, data_num]
            x = self.data                                      # [data_dim, data_num]
            nominator = (tau_k * (x - mu_k)) @ (x - mu_k).T    # [data_dim, data_dim]
            denominator = taus[k, :].sum()
            covs[:, :, k] = nominator/denominator
        self.covs = covs

    def train(self):
        """
        Train GMM
        """
        self._initialization()
        print(f"{arrow.now()}: finish initialization")
        for it in range(self.iternum):
            self.taus = self.Estep()
            print(f"{arrow.now()}: {it}th Estep")
            self.Mstep()
            print(f"{arrow.now()}: {it}th Mstep")
    
    def EvaluateMu(self):
        """
        Evaluate mean plot
        """
        # pay attention to the shape of imdata when reshape it
        for k in range(self.C):
            data     = self.mus[:, k]
            imdata   = np.reshape(data, 
                newshape=(28, 28), 
                order='F')                                 # [28, 28]
            fig = plt.figure()    # Baseline model KMeans
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=1).fit(data.T)
    idxs   = kmeans.labels_

    preds = []
    for idx in idxs:
        if idx == 1:
            preds.append(6.)
        else:
            preds.append(2.)
    preds = np.asarray(preds)

    correct2 = 0
    wrong2   = 0
    correct6 = 0
    wrong6   = 0
    for idx, label in enumerate(label):
        if label == 2.:
            if preds[idx] == label:
                correct2 += 1
            else:
                wrong2 += 1
        elif label == 6.:
            if preds[idx] == label:
                correct6 += 1
            else:
                wrong6 += 1
    mcr2 = wrong2/(wrong2+correct2)
    mcr6 = wrong6/(wrong6+correct6)
    print(f"Kmeans: label2 : correct:{correct2} wrong:{wrong2} mcr:{mcr2}")
    print(f"Kmeans: label6 : correct:{correct6} wrong:{wrong6} mcr:{mcr6}")  
            ax  = fig.add_subplot(111)
            ax.imshow(imdata)
            plt.show()
    
    def EvaluatePi(self):
        """
        Evaluate Pic
        """
        print(self.pis)

    def EvaluatePredError(self):
        """
        Evaluate prediction error
        """
        tau  = self.taus               # [C, data_num]
        idxs = np.argmax(tau, axis=0)  # [data_num,]
        preds = []
        for idx in idxs:
            if idx == 1:
                preds.append(2.)
            else:
                preds.append(6.)
        preds = np.asarray(preds)
        print(f"prediction:{preds}")
        print(f"truth:{self.label}")

        correct2 = 0
        wrong2   = 0
        correct6 = 0
        wrong6   = 0
        for idx, label in enumerate(self.l    # Baseline model KMeans
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=1).fit(data.T)
    idxs   = kmeans.labels_

    preds = []
    for idx in idxs:
        if idx == 1:
            preds.append(6.)
        else:
            preds.append(2.)
    preds = np.asarray(preds)

    correct2 = 0
    wrong2   = 0
    correct6 = 0
    wrong6   = 0
    for idx, label in enumerate(label):
        if label == 2.:
            if preds[idx] == label:
                correct2 += 1
            else:
                wrong2 += 1
        elif label == 6.:
            if preds[idx] == label:
                correct6 += 1
            else:
                wrong6 += 1
    mcr2 = wrong2/(wrong2+correct2)
    mcr6 = wrong6/(wrong6+correct6)
    print(f"Kmeans: label2 : correct:{correct2} wrong:{wrong2} mcr:{mcr2}")
    print(f"Kmeans: label6 : correct:{correct6} wrong:{wrong6} mcr:{mcr6}")  abel):
            if label == 2.:
                if preds[idx] == label:
                    correct2 += 1
                else:
                    wrong2 += 1
            elif label == 6.:
                if preds[idx] == label:
                    correct6 += 1
                else:
                    wrong6 += 1
        mcr2 = wrong2/(wrong2+correct2)
        mcr6 = wrong6/(wrong6+correct6)
        print(f"GMM: label2 : correct:{correct2} wrong:{wrong2} mcr:{mcr2}")
        print(f"GMM: label6 : correct:{correct6} wrong:{wrong6} mcr:{mcr6}")  

    def ShowLoglik(self):
        """
        Plot the loglikelihood
        """
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(self.logliks)
        plt.show()

    def ShowImage(self, data):
        """
        data: [784, ]
        """
        # pay attention to the shape of imdata when reshape it
        imdata   = np.reshape(data, 
            newshape=(28, 28), 
            order='F')                                 # [28, 28]
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.imshow(imdata)
        plt.show()

def Kmeans(data, label):
    # Baseline model KMeans    # Baseline model KMeans
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=1).fit(data.T)
    idxs   = kmeans.labels_

    preds = []
    for idx in idxs:
        if idx == 1:
            preds.append(6.)
        else:
            preds.append(2.)
    preds = np.asarray(preds)

    correct2 = 0
    wrong2   = 0
    correct6 = 0
    wrong6   = 0
    for idx, label in enumerate(label):
        if label == 2.:
            if preds[idx] == label:
                correct2 += 1
            else:
                wrong2 += 1
        elif label == 6.:
            if preds[idx] == label:
                correct6 += 1
            else:
                wrong6 += 1
    mcr2 = wrong2/(wrong2+correct2)
    mcr6 = wrong6/(wrong6+correct6)
    print(f"Kmeans: label2 : correct:{correct2} wrong:{wrong2} mcr:{mcr2}")
    print(f"Kmeans: label6 : correct:{correct6} wrong:{wrong6} mcr:{mcr6}")  
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=1).fit(data.T)
    idxs   = kmeans.labels_

    preds = []
    for idx in idxs:
        if idx == 1:
            preds.append(6.)
        else:
            preds.append(2.)
    preds = np.asarray(preds)

    correct2 = 0
    wrong2   = 0
    correct6 = 0
    wrong6   = 0
    for idx, label in enumerate(label):
        if label == 2.:
            if preds[idx] == label:
                correct2 += 1
            else:
                wrong2 += 1
        elif label == 6.:
            if preds[idx] == label:
                correct6 += 1
            else:
                wrong6 += 1
    mcr2 = wrong2/(wrong2+correct2)
    mcr6 = wrong6/(wrong6+correct6)
    print(f"Kmeans: label2 : correct:{correct2} wrong:{wrong2} mcr:{mcr2}")
    print(f"Kmeans: label6 : correct:{correct6} wrong:{wrong6} mcr:{mcr6}")  


def main():
    np.random.seed(2)
    data  = np.loadtxt("data/data.dat")    # [784, 1990]
    label = np.loadtxt("data/label.dat")   # [1900, ]

    # GMM 
    mymodel = GaussianMixtureModel(data, label, 2, 20, 50)
    mymodel.ShowImage(data[:, 0])
    mymodel.ShowImage(data[:, -1])
    mymodel.train()
    mymodel.EvaluateMu()
    mymodel.EvaluatePi()
    mymodel.ShowLoglik()
    mymodel.EvaluatePredError()
    
    # Kmeans 
    Kmeans(data, label)

if __name__ == '__main__':
    main()
    