import json, pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
r=pd.read_csv("results/all_runs.csv"); a=json.load(open("results/artifacts.json"))
order=list(dict.fromkeys(r.model)); C=dict(zip(order,["#7f7f7f","#c58b2a","#8e5ea2","#3a8f6b","#2f6db5","#d1495b"]))
g=r.groupby("model",sort=False)
# Fig1 loss curves
fig,ax=plt.subplots(1,2,figsize=(9,3.4),sharey=True)
for n in order:
    h=a["histories"][n]; ax[0].plot(h["loss"],color=C[n],label=n,lw=1.6); ax[1].plot(h["val_loss"],color=C[n],lw=1.6)
ax[0].set_title("Training loss"); ax[1].set_title("Validation loss")
for x in ax: x.set_xlabel("Epoch")
ax[0].set_ylabel("Binary cross-entropy"); ax[0].legend(fontsize=7,frameon=False); plt.tight_layout(); plt.savefig("fig1_loss.png"); plt.close()
# Fig2 metric bars
ms=["accuracy","precision","recall","f1","auc"]; w=0.13
fig,ax=plt.subplots(figsize=(9,3.6))
for i,n in enumerate(order):
    m=g.get_group(n)[ms].mean(); s=g.get_group(n)[ms].std()
    ax.bar(np.arange(5)+i*w,m,w,yerr=s,color=C[n],label=n,capsize=1.5,error_kw={"lw":.7})
ax.set_xticks(np.arange(5)+w*2.5); ax.set_xticklabels(["Accuracy","Precision","Recall","F1","ROC-AUC"]); ax.set_ylim(0.4,0.9)
ax.set_ylabel("Test score (mean ± SD, 3 seeds)"); ax.legend(ncol=3,fontsize=7,frameon=False,loc="upper left"); plt.tight_layout(); plt.savefig("fig2_metrics.png"); plt.close()
# Fig3 ROC + confusion
y=np.array(a["y_te"]); fig,ax=plt.subplots(1,2,figsize=(9,3.6))
for n in ["Baseline SGD","Adagrad","Adam","Adam (tuned)"]:
    f,t,_=roc_curve(y,a["probs"][n]); ax[0].plot(f,t,color=C[n],label=n,lw=1.6)
ax[0].plot([0,1],[0,1],"k:",lw=.8); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate"); ax[0].legend(fontsize=8,frameon=False); ax[0].set_title("ROC curves (seed 1)")
cm=np.array(a["cms"]["Adam (tuned)"]); ax[1].imshow(cm,cmap="Blues")
for i in range(2):
    for j in range(2): ax[1].text(j,i,cm[i,j],ha="center",va="center",fontsize=14,color="white" if cm[i,j]>cm.max()/2 else "black")
ax[1].set_xticks([0,1]); ax[1].set_xticklabels(["Stay","Churn"]); ax[1].set_yticks([0,1]); ax[1].set_yticklabels(["Stay","Churn"])
ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Actual"); ax[1].set_title("Adam (tuned): confusion matrix"); ax[1].spines[:].set_visible(False)
plt.tight_layout(); plt.savefig("fig3_roc_cm.png"); plt.close()
# EDA fig
d=pd.read_csv("teleconnect.csv"); d["ch"]=(d.Churn=="Yes")
fig,ax=plt.subplots(1,3,figsize=(9,3))
for x,col,t in zip(ax,["Contract","InternetService","PaymentMethod"],["Contract","Internet service","Payment method"]):
    s=d.groupby(col).ch.mean().sort_values(); x.barh([k.replace(" (automatic)","") for k in s.index],s.values*100,color="#2f6db5"); x.set_title(t); x.set_xlabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("fig0_eda.png"); plt.close()
print(d.groupby("Contract").ch.mean().round(3).to_dict(), d.groupby("InternetService").ch.mean().round(3).to_dict())
print(d.groupby(pd.cut(d.tenure,[-1,12,24,48,72])).ch.mean().round(3).to_dict())
