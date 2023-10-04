from PIL import Image
import clip
import torch
from utils import ModelWrapper
from tqdm import tqdm
import argparse
import os
from torchvision.datasets import ImageFolder
from utils import ModelWrapper, maybe_dictionarize_batch, cosine_lr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('~/tmp123'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--custom-template", action="store_true", default=False,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--warmup-length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--model",
        default='ViT-B/32',
        help='Model to use -- you can try another like ViT-L/14'
    )
    parser.add_argument(
        "--name",
        default='finetune_cp',
        help='Filename for the checkpoints.'
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the weights.",
    )
    parser.add_argument(
        "--obj",
        type=str,
        help="syringe or vial",
    )
    parser.add_argument(
        "--drawup",
        action="store_true",
        default=False,
        help="true/false",
    )
    parser.add_argument(
        "--version",
        type=int    
    )
    parser.add_argument(
        "--tag",
        type=str
    )
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    DEVICE = 'cuda'
    print (args.obj)
    print (args.version)
    if args.obj == 'syringe':
        if args.version==1:
            classnames = ['cisatracurium', 'dexamethasone', 'ephedrine', 'epinephrine', 'etomidate', 'fentanyl', 'glycopyrrolate', 'hydromorphone', 'ketamine', 'ketorolac', 'lidocaine', 'midazolam', 'neostigmine', 'ondansetron', 'phenylephrine', 'propofol', 'rocuronium', 'succinylcholine', 'sugammadex', 'vecuronium']
        elif args.version==2:
            classnames=["dexamethasone","fentanyl","hydromorphone","ketamine","ketorolac","lidocaine","midazolam","neostigmine","ondansetron","propofol","rocuronium","sugammadex","vecuronium"]
        elif args.version==3:
            classnames=["background","dexamethasone","fentanyl","hydromorphone","ketamine","ketorolac","lidocaine","midazolam","neostigmine","ondansetron","propofol","rocuronium","sugammadex","vecuronium"]
        elif args.version==4:
            classnames = ['background','cisatracurium', 'dexamethasone', 'ephedrine', 'epinephrine', 'etomidate', 'fentanyl', 'glycopyrrolate', 'hydromorphone', 'ketamine', 'ketorolac', 'lidocaine', 'midazolam', 'neostigmine', 'ondansetron', 'phenylephrine', 'propofol', 'rocuronium', 'succinylcholine', 'sugammadex', 'vecuronium']
    elif args.obj == 'vial' or args.obj=='vial-label':
        if args.version==1:
            classnames = ['Amiodarone', 'Background', 'Dexamethasone', 'Etomidate', 'Fentanyl', 'Glycopyrrolate', 'Hydromorphone', 'Ketamine', 'Ketorolac', 'Lidocaine', 'Magnesium Sulfate', 'Midazolam', 'Neostigmine', 'Ondansetron', 'Propofol', 'Rocuronium', 'Vasopressin', 'Vecuronium']
        elif args.version==2:
            classnames=["Dexamethasone","Etomidate","Fentanyl","Glycopyrrolate","Hydromorphone","Ketorolac","Lidocaine","Midazolam","Neostigmine","Ondansetron","Propofol","Rocuronium","Vasopressin"]
        elif args.version==3:
            classnames=["Background","Dexamethasone","Etomidate","Fentanyl","Glycopyrrolate","Hydromorphone","Ketorolac","Lidocaine","Midazolam","Neostigmine","Ondansetron","Propofol","Rocuronium","Vasopressin"]
    fout = open ('results.txt','a+')
    
    NUM_CLASSES = len(classnames)

    base_model, preprocess = clip.load(args.model, 'cuda', jit=False)
    
    print ("finished loading model")
    model = ModelWrapper(base_model, base_model.visual.output_dim, NUM_CLASSES, normalize=True)
    for p in model.parameters():
        p.data = p.data.float()

    saved_model_path = args.weights
    print (saved_model_path)
    print (args.version)
    print (classnames)
    print (NUM_CLASSES)
    ll = torch.load(saved_model_path, map_location='cpu') 
    model.load_state_dict(ll)
    for p in model.parameters():
        p.data = p.data.float()
    print ("finished loading weights")

    model = model.cuda()
    devices = [x for x in range(torch.cuda.device_count())]
    model = torch.nn.DataParallel(model,  device_ids=devices)

    loss_fn = torch.nn.CrossEntropyLoss()

    model.eval()
    
    if args.drawup:
        fs=sorted(os.listdir('/gscratch/intelligentsystems/justin/real_vids/'))
        drawups=[]
        for i in fs:
            if '.mp4' in i in i:
                drawups.append (i[:-4])
        drawups=[i.replace('_Clip','') for i in drawups]
        files = drawups
    else:
        files = range(1,236)

    for e in files:
        exp='exp'+str(e)
        if args.drawup:
            dout='/gscratch/cse/jucha/logs_bw_real_'+str(args.version)+'/log_'+args.obj+'_'+exp+'.txt'
        else:
            dout='/gscratch/cse/jucha/logs_extra_v'+str(args.version)+'/log_'+args.obj+'_'+exp+'.txt'
        if os.path.exists(dout) and os.stat(dout).st_size != 0:
            continue
        fout2 = open (dout,'w')

        if args.drawup:
            if args.obj == 'syringe':
                valdir = '/gscratch/cse/jucha/syringe_real/detect/'+exp+'/crops/'
            elif args.obj == 'vial' or args.obj=='vial-label':
                valdir = '/gscratch/cse/jucha/vial_real/detect/'+exp+'/crops/'
        else:
            if args.obj == 'syringe':
                valdir = '/gscratch/cse/jucha/syringe_extra/detect/'+exp+'/crops/'
            elif args.obj == 'vial' or args.obj=='vial-label':
                valdir = '/gscratch/cse/jucha/vial_extra/detect/'+exp+'/crops/'
        # valdir = args.data_location
        print (valdir)
        print ('valdir ',valdir,os.path.exists(valdir))
        if not os.path.exists(valdir):
            print ('continue ',valdir)
            continue
        test_dataset = ImageFolder(valdir, transform=preprocess)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

        preds_out=[]
        labels_out=[]
        counter = {}
        idx=0
        correct=0
        with torch.no_grad():
            print('*'*80)
            print('Starting eval')
            correct, count = 0.0, 0.0
            pbar = tqdm(test_loader)
            correct_fname=[]
            wrong_fname=[]
            correct_vals=[]
            wrong_vals=[]
            correct_logit=[]
            wrong_logit=[]
            for batch in pbar:
                batch = maybe_dictionarize_batch(batch)

                inputs, labels = batch['images'].to(DEVICE), batch['labels'].to(DEVICE)

                logits = model(inputs)
                #prob = logits.topk(3,dim=1)[0]
                #prob = torch.nn.functional.softmax(logits, dim=1)
                #prob = prob.topk(3)[0]
                #print (prob)

                pred = logits.topk(3,dim=1)[1]
                #print (pred)
                
                if len(pred.cpu().numpy())>0:
                    logits = list(logits.cpu().numpy().squeeze())
                    preds = list(pred.cpu().numpy().squeeze())
                    #probs = list(prob.cpu().numpy().squeeze())
                else:
                    preds=[]
                    probs=[]

                for k in range(len(preds)):
                    #print (idx,len(test_loader.dataset.samples))
                    if idx>=len(test_loader.dataset.samples):
                        break
                    fname=test_loader.dataset.samples[idx][0]
                    fname_conf = fname.split('/')
                    #print ('fname_conf ',fname_conf)
                    detected_class=fname_conf[-2]
                    #print ('detected class ', detected_class)
                    full_txt_name = '/'.join(fname_conf[0:-3])+"/labels/"+fname_conf[-1][:-4]+".txt"
                    vlabel = "Vial"
                    if args.obj=='vial':
                        vlabel='Vial'
                    elif args.obj=='vial-label':
                        vlabel='Vial Label'
                    if (detected_class==vlabel or detected_class=='Label') and os.path.exists(full_txt_name):
                        #print ('full_txt_name ', full_txt_name)
                        conf_file=open(full_txt_name).read().split('\n')
                        conf=0
                        for line2 in conf_file:
                            #print (line2)
                            if len(line2)>1 and ((detected_class=='Vial' and line2[0]=='0') or (detected_class=='Label' and line2[0]=='1')):
                                elts=line2.split(' ')
                                conf=elts[-1]
                                break
                        #try:
                        logits_out = logits[k].tolist()
                        #print ('logit ',logits_out)
                        if isinstance(logits_out,float):
                            logits_out=[logits_out]
                        l=''
                        for i in logits_out:
                            l+=str(i)+","

                        #print (logits_out)
                        fout2.write(str(fname)+","+str(conf)+","+l+"\n")
                        fout2.flush()
                        #fout2.write(str(fname)+","+conf+","+str(probs[k][0])+","+str(probs[k][1])+","+str(probs[k][2])+","+str(classnames[preds[k][0]])+","+str(classnames[preds[k][1]])+","+str(classnames[preds[k][2]])+"\n")
                        #except e:
                            #print (e)
                            #print ("error")
                            #print (preds[k])
                            #print (classnames)
                    idx+=1
            fout2.flush()
            #break
        fout2.close()
    



















