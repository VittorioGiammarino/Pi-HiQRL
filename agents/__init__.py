from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gciql import GCIQLAgent
from agents.gcivl import GCIVLAgent
from agents.hiql import HIQLAgent
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.pi_qrl import PI_QRL_Agent
from agents.pi_qrl_hi import PI_QRL_HI_Agent 
from agents.pi_qrl_lam import PI_QRL_LAM_Agent
from agents.pi_qrl_lam_hi import PI_QRL_LAM_HI_Agent
from agents.pi_hiqrl import PI_HIQRL_Agent
from agents.hiqrl import HIQRL_Agent

agents = dict(
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gciql=GCIQLAgent,
    gcivl=GCIVLAgent,
    hiql=HIQLAgent,
    qrl=QRLAgent,
    sac=SACAgent,
    pi_qrl=PI_QRL_Agent,
    pi_qrl_hi=PI_QRL_HI_Agent,
    pi_qrl_lam = PI_QRL_LAM_Agent,
    pi_qrl_lam_hi = PI_QRL_LAM_HI_Agent,
    pi_hiqrl = PI_HIQRL_Agent,
    hiqrl=HIQRL_Agent,
)
