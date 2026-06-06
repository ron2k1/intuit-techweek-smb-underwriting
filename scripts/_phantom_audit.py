import numpy as np, pandas as pd

APR=0.35; TERM_DAYS=60; FEE=0.03
INT=APR*TERM_DAYS/365.0
PAID=FEE+INT
GROSS=1.0+INT
DRAW=GROSS/TERM_DAYS

def npv_repaid(a): return a*PAID
def npv_default(a,t,rec,cap):
    fee=FEE*a
    clip = np.clip(t-1,0,None) if cap is None else np.clip(t-1,0,cap)
    return fee+DRAW*a*clip+rec-a
def realized(df,cap):
    a=df.requested_amount.to_numpy(float)
    d=df.default_flag.fillna(0).to_numpy(float)
    t=df.days_to_default.fillna(0).to_numpy(float)
    rec=df.final_recovered_amount.fillna(0).to_numpy(float)
    return np.where(d==1, npv_default(a,t,rec,cap), npv_repaid(a))

# validation is the only labeled split
val=pd.read_csv('data/csv-files/validation.csv',
    usecols=['applicant_id','requested_amount','default_flag','days_to_default','final_recovered_amount'])
sub=pd.read_csv('outputs/submission/submission_A_decisions.csv')[['applicant_id','decision']]
df=val.merge(sub,on='applicant_id',how='inner')
print('validation rows merged with submission_A:',len(df))

df['npv_unc']=realized(df,None)
df['npv_cap']=realized(df,TERM_DAYS-1)

appr=df[df.decision==1]
print('\n--- Steven approved on validation:',len(appr),'of',len(df))
def fmt(x): return f"${x:,.0f}"

rows=[]
for label,sel in [('Steven policy (approved)',df.decision==1),('Approve-all baseline',df.applicant_id.notna())]:
    s=df[sel]
    u=s.npv_unc.sum(); c=s.npv_cap.sum()
    rows.append((label,len(s),u,c,u-c))

print('\n{:<28} {:>7} {:>16} {:>16} {:>14}'.format('policy','n','NPV uncapped','NPV capped','phantom delta'))
for label,n,u,c,d in rows:
    print('{:<28} {:>7} {:>16} {:>16} {:>14}'.format(label,n,fmt(u),fmt(c),fmt(d)))

# phantom drivers among approved
appr_def=appr[appr.default_flag==1]
appr_post60=appr_def[appr_def.days_to_default>TERM_DAYS]
print('\nApproved defaults:',len(appr_def),'| approved post-day-60 defaults:',len(appr_post60))
ph=(appr.npv_unc-appr.npv_cap)
print('approved loans with phantom>0:',(ph>1e-6).sum(),'(should equal post-60 approved defaults)')
print('total phantom on approved:',fmt(ph.sum()))
if len(appr_post60):
    perloan=(appr_post60.npv_unc-appr_post60.npv_cap)
    print('per-loan phantom on post-60 approved defaults: mean',fmt(perloan.mean()),
          'min',fmt(perloan.min()),'max',fmt(perloan.max()))
    print('their days_to_default range:',int(appr_post60.days_to_default.min()),'-',int(appr_post60.days_to_default.max()))

# all post-60 defaults in val (regardless of decision)
all_post60=df[(df.default_flag==1)&(df.days_to_default>TERM_DAYS)]
print('\nAll post-60 defaults in val:',len(all_post60),
      '| of total defaults',int((df.default_flag==1).sum()),
      f'= {100*len(all_post60)/(df.default_flag==1).sum():.1f}%')

# competitors
print('\n--- Competitors on validation (uncapped vs capped) ---')
import os
for who in ['ronil','ayush','Abhimanyu']:
    p=f'comparison/worktrees/{who}/submissions/submission_A_decisions.csv'
    if not os.path.exists(p): continue
    c=pd.read_csv(p)
    col='decision' if 'decision' in c.columns else [x for x in c.columns if 'dec' in x.lower()][0]
    m=val.merge(c[['applicant_id',col]],on='applicant_id',how='inner')
    if not len(m):
        print(who,'no id overlap with validation'); continue
    m['nu']=realized(m,None); m['nc']=realized(m,TERM_DAYS-1)
    a=m[m[col]==1]
    print(f'{who:<10} approved {len(a):>4}/{len(m)}  uncapped {fmt(a.nu.sum()):>14}  capped {fmt(a.nc.sum()):>14}  delta {fmt(a.nu.sum()-a.nc.sum())}')
