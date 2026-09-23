import re, glob, os, sys, numpy as np, pandas as pd, matplotlib
from matplotlib.ticker import FuncFormatter, MaxNLocator
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import figure_palette as PAL
PAL.apply_palette(plt)   # series that name no colour land on the palette
# Code_Submission root: the parent of this script's folder, or argv[2].
CS=os.path.abspath(sys.argv[2]) if len(sys.argv)>2 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT=sys.argv[1]; os.makedirs(OUT,exist_ok=True)
TW=522/72.27; DPI=600
plt.rcParams.update(PAL.house_rcparams())
def save(fig,stem):
    fig.savefig(f"{OUT}/{stem}.png",dpi=DPI); fig.savefig(f"{OUT}/{stem}.pdf"); plt.close(fig); print("wrote",stem)
CH=["Generation--load mismatch","Seller--buyer nodal price decoupling","Buyer load--price intensification","Seller generation--price cannibalization"]
CHT={"Generation--load mismatch":"Generation–load mismatch","Seller--buyer nodal price decoupling":"Seller–buyer nodal price\ndecoupling","Buyer load--price intensification":"Buyer load–price\nintensification","Seller generation--price cannibalization":"Seller generation–price\ncannibalization"}
# ---------- risk-averse family ----------
ra=pd.read_csv(CS+"/simulation_mutation_averse/Output files (Risk Averse, Mutation, Verified)/Risk_Averse_Mutation_Analysis/tables/RA_Mutation_Paired_Baseline_Changes_Long.csv",low_memory=False)
ra=ra[ra.scenario_type=="mutation"]
fam=ra.mutation_family_normalized.astype(str)
print("family labels in RA table:",sorted(fam.unique()))
famap={}
for f in fam.unique():
    fl=f.lower()
    if "generation" in fl and "load" in fl or "shape" in fl: famap[f]=CH[0]
    elif "nodal" in fl or "basis" in fl: famap[f]=CH[1]
    elif "load" in fl and "price" in fl: famap[f]=CH[2]
    elif "cannibal" in fl: famap[f]=CH[3]
ra["channel"]=fam.map(famap); ra=ra.dropna(subset=["channel"])
ra["shift"]=ra.target_shift_signed.astype(float); ra["ai"]=ra.target_shift_abs.astype(float).rank(method="dense").astype(int)-1
SERIES=[("joint_low","joint-low ($\\lambda^S=\\lambda^B=0.253$)","o",PAL.BLUE),("joint_medium","joint-medium ($\\lambda^S=\\lambda^B=0.524$)","s",PAL.AMBER)]
# Height ratio 0.58 (was 0.62): at 0.62 Figures 7 and 8 each took a page top and the
# manuscript's Conclusion page was left 5 lines short; at 0.58 the two fit one page
# together and every page foot in the compiled manuscript is within 8 pt (measured).
RA_ASPECT=0.58
def ra_fig(col,ylabel,stem,width,fixonly=False,scale=1.0):
    fig,axes=plt.subplots(2,2,figsize=(width,width*RA_ASPECT),dpi=DPI); span={}; arts={}
    for ax,ch in zip(axes.ravel(),CH):
        span[ax]=0.0
        for rl,lab,mk,c in SERIES:
            d=ra[(ra.channel==ch)&(ra.risk_label==rl)]
            if fixonly: d=d[d.fix_to_fix.astype(str).str.lower().eq("true")]
            g=d.groupby("ai")[col]; med=g.median()*scale; q1=g.quantile(.25)*scale; q3=g.quantile(.75)*scale; x=med.index.values
            band=ax.fill_between(x,q1.values,q3.values,color=c,alpha=0.18,linewidth=0)
            span[ax]=max(span[ax],float(np.nanmax(np.abs(np.r_[med.values,q1.values,q3.values]))))
            line,=ax.plot(x,med.values,marker=mk,color=c,label=lab); arts[(ax,rl)]=(band,line,med.values,q1.values,q3.values)
        ticks=d.groupby("ai")["shift"].first()
        ax.set_xticks(ticks.index.values); ax.set_xticklabels([f"{abs(v):.2f}" for v in ticks.values]); ax.set_xlim(-0.3,2.3)
        ax.axhline(0,color="0.5",linewidth=0.5,zorder=0); ax.grid(True); ax.set_title(CHT[ch],pad=5)
        for sp in ax.spines.values(): sp.set_linewidth(0.6)
    # When the two settings give identical medians and quartiles in every panel (Figure 8),
    # one series would hide the other: draw one and say so in the legend.
    same=all(all(np.allclose(arts[(ax,SERIES[0][0])][i],arts[(ax,SERIES[1][0])][i],equal_nan=True) for i in (2,3,4)) for ax in axes.ravel())
    if same:
        for ax in axes.ravel():
            arts[(ax,SERIES[0][0])][0].remove(); arts[(ax,SERIES[0][0])][1].remove()
            arts[(ax,SERIES[1][0])][1].set_label("joint-low ($\\lambda^S=\\lambda^B=0.253$) and joint-medium ($\\lambda^S=\\lambda^B=0.524$), identical")
        print("identical series, one drawn:",stem)
    # The autoscaled limits cover the bands as well as the medians, so no interquartile
    # band runs off the frame (the strike-price bands reach -5 $/MWh under zero medians).
    # A panel that is zero throughout, band included, gets a small symmetric range with
    # three ticks, so the scale is readable. Tick labels then share one decimal width.
    gmax=max(abs(v) for ax in axes.ravel() for ln in ax.get_lines() for v in ln.get_ydata() if v==v) or 1.0
    for ax in axes.ravel():
        if span[ax]<1e-9: ax.set_ylim(-0.15*gmax,0.15*gmax); ax.yaxis.set_major_locator(MaxNLocator(nbins=4,symmetric=True))
    def _dec(v):
        for k in range(4):
            if abs(round(v,k)-v)<1e-9: return k
        return 3
    k=max([_dec(t) for ax in axes.ravel() for t in ax.get_yticks() if ax.get_ylim()[0]<=t<=ax.get_ylim()[1]] or [0])
    fmt=FuncFormatter(lambda v,_: f"{(0.0 if abs(v)<1e-12 else v):.{k}f}".replace("-","\u2212"))
    for ax in axes.ravel(): ax.yaxis.set_major_formatter(fmt)
    for ax in axes[1]: ax.set_xlabel("Shift size")
    for ax in axes[:,0]: ax.set_ylabel(ylabel)
    h,l=axes[0,0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=1 if same else 2,frameon=True,bbox_to_anchor=(0.5,-0.01))
    fig.tight_layout(rect=(0,0.06,1,1),pad=0.4,w_pad=1.2,h_pad=1.0); save(fig,stem)
ra_fig("delta_fixed_volume_mw","Δ contracted volume $q$ (MW)","Fig 7. Contracted volume changes",TW,fixonly=True)
ra_fig("delta_strike_price_mwh","Δ strike price ($/MWh)","Fig 8. Strike-price changes under risk aversion",TW)
ra_fig("delta_delivered_volume_proxy_mw","Δ mean delivered volume (MW)","Fig A7. Delivered-volume changes under risk aversion",TW)
ra_fig("delta_seller_metric","Δ seller exposure index\n(\\$ million)","Fig A8. Seller exposure changes under risk aversion",TW,scale=1e-6)
ra_fig("delta_buyer_metric","Δ buyer exposure index\n(\\$ million)","Fig A9. Buyer exposure changes under risk aversion",TW,scale=1e-6)
ra_fig("delta_buyer_participation_slack","Δ buyer participation slack\n(\\$ million)","Fig A10. Buyer participation slack changes",TW,scale=1e-6)
# ---------- channel-bank lambda data for A4-A6 ----------
B="Baseline__No_Mutation__Verified"
ref=pd.read_csv(CS+"/simulation_mutation/Output files (Risk Neutral, Mutation, Verified)/Simulation_Best_Solutions_All_Matches.csv",low_memory=False); ref=ref[ref.scenario_name==B].set_index("match_id").sort_index()
def stats(x):
    fix=x.profile_type=="Fix"
    return dict(ident=float(((x.profile_type==ref.profile_type)&(x.strike_price_mwh==ref.strike_price_mwh)&(x.volume_mw.fillna(-1)==ref.volume_mw.fillna(-1))).mean()),ppa=float(x.profile_type.isin(["Fix","AsC","AsG"]).mean()),fix=float(fix.mean()),asc=float((x.profile_type=="AsC").mean()),strike=float(x.strike_price_mwh.mean()),vol=float(x.loc[fix,"volume_mw"].mean()))
lam={"seller":{0.0:stats(ref)},"buyer":{0.0:stats(ref)},"joint":{0.0:stats(ref)}}
for d in sorted(glob.glob(CS+"/simulation_mutation_averse/Output files (Risk Averse, Mutation, Verified)/0*/")):
    m=re.search(r"lambdaS_(\dp\d+)__lambdaB_(\dp\d+)",d); ls,lb=[float(v.replace("p",".")) for v in m.groups()]
    f=d+"Simulation_Best_Solutions_All_Matches.csv"
    if not os.path.exists(f): continue
    x=pd.read_csv(f,low_memory=False); x=x[x.scenario_name==B].set_index("match_id").sort_index()
    axis="seller" if lb==0 else ("buyer" if ls==0 else "joint"); lam[axis][max(ls,lb)]=stats(x)
sens=pd.read_csv(CS+"/sensitivity_baseline/Output files (Sensitivity, No Mutation, Verified)/Sensitivity_Case_Level_Summary.csv").set_index("case_id")
chg=pd.read_csv(CS+"/sensitivity_baseline/Output files (Sensitivity, No Mutation, Verified)/Sensitivity_Change_Summary_By_Case.csv").set_index("case_id")
print("check: sensitivity BASE volume_mean",sens.loc["BASE","volume_mean"],"| channel ref Fix-only mean vol",round(stats(ref)["vol"],1),"| ref strike mean",round(stats(ref)["strike"],2),"vs",sens.loc["BASE","strike_mean"])
base=sens.loc["BASE"]
alpha={"Volume residual scale":[("VOL_075",0.75),("BASE",1.0),("VOL_125",1.25)],"Price residual scale":[("PRICE_075",0.75),("BASE",1.0),("PRICE_125",1.25)]}
LSER=[("seller","Seller risk aversion $\\lambda^S$","o"),("buyer","Buyer risk aversion $\\lambda^B$","s"),("joint","Joint risk aversion $\\lambda^S=\\lambda^B$","^")]
def a_fig(stem,rows):
    fig,axes=plt.subplots(2,2,figsize=(TW,TW*0.60),dpi=DPI)
    for r,(ylabel,lkey,akey,pct) in enumerate(rows):
        ax=axes[r,0]
        for axis,lab,mk in LSER:
            ks=sorted(lam[axis]); ys=[lkey(lam[axis][k]) for k in ks]
            ax.plot(ks,ys,marker=mk,markerfacecolor="white",markeredgewidth=0.9,label=lab)
        ax.set_ylabel(ylabel)
        ax2=axes[r,1]
        for name,cases in alpha.items():
            xs=[a for _,a in cases]; ys=[akey(sens.loc[c],chg.loc[c]) for c,_ in cases]
            ax2.plot(xs,ys,marker="D" if "Volume" in name else "v",markerfacecolor="white",
                     markeredgewidth=0.9,color=PAL.WINE if "Volume" in name else PAL.INDIGO,label=name)
        for a in (ax,ax2):
            PAL.tidy(a)
            if pct: a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0,decimals=(1 if pct=="signed" else 0)))
            if pct is True: a.set_ylim(-0.03,1.03)
        if pct=="signed":
            for a in (ax,ax2): a.axhline(0,color=PAL.MUTED,linewidth=0.6,linestyle="--")
        # Where the two columns of a row already span the same range, the right
        # column repeats the tick labels for nothing; Figure 3 drops them too.
        if np.allclose(ax.get_ylim(),ax2.get_ylim()):
            ax2.set_yticklabels([]); ax2.spines["left"].set_visible(False); ax2.tick_params(axis="y",length=0)
    axes[0,0].set_title("Risk aversion",pad=4); axes[0,1].set_title("Residual scale",pad=4)
    axes[1,0].set_xlabel("Risk-aversion weight"); axes[1,1].set_xlabel("Residual scale $\\alpha$")
    h1,l1=axes[0,0].get_legend_handles_labels(); h2,l2=axes[0,1].get_legend_handles_labels()
    leg=fig.legend(h1+h2,l1+l2,loc="lower left",ncol=5,bbox_to_anchor=(0.010,0.905),
               bbox_transform=fig.transFigure,columnspacing=0.7,handlelength=1.2,handletextpad=0.35)
    PAL.fit_legend(fig,leg)
    fig.subplots_adjust(left=0.090,right=0.995,bottom=0.105,top=0.840,wspace=0.16,hspace=0.30)
    save(fig,stem)
a_fig("Fig A4. Complete-contract agreement and PPA selection",[("Complete-contract agreement",lambda s:s["ident"],lambda s,c:c["share_same_full_decision"],True),("PPA-selection share",lambda s:s["ppa"],lambda s,c:s["ppa_share"],True)])
a_fig("Fig A5. Contract term sensitivity",[("Mean strike price change",lambda s:s["strike"]/lam["seller"][0.0]["strike"]-1,lambda s,c:s["strike_mean"]/base["strike_mean"]-1,"signed"),("Mean contracted volume change",lambda s:s["vol"]/lam["seller"][0.0]["vol"]-1,lambda s,c:s["volume_mean"]/base["volume_mean"]-1,"signed")])
a_fig("Fig A6. PPA structure share sensitivity",[("Fixed-Volume selection share",lambda s:s["fix"],lambda s,c:s["fix_share"],True),("As-Consumed selection share",lambda s:s["asc"],lambda s,c:s["asc_share"],True)])
