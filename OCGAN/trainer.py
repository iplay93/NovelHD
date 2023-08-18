def train_ae(args,trainloader, enc, dec, optimizer_en, optimizer_de, criterion, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    end = time.time()
    

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        u = np.random.uniform(-1, 1, (args.train_batch, 288, 1, 1))   
        l2 = torch.from_numpy(u).float()

        n = torch.randn(args.train_batch, 1, 28, 28).cuda()
        l1 = enc(inputs + n)
        del1 = dec(l1)
  

        loss = criterion(del1,inputs)


        losses.update(loss.item(), inputs.size(0))


        enc.zero_grad()
        dec.zero_grad()

        loss.backward()

        optimizer_en.step()
        optimizer_de.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}  '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()
    return losses.avg

def train(args,trainloader,enc, dec,cl,disc_l,disc_v,
                    optimizer_en, optimizer_de,optimizer_c,optimizer_dl,optimizer_dv,optimizer_l2,
                    criterion_ae, criterion_ce,Tensor, epoch, use_cuda):
    # switch to train mode
    enc.train()
    dec.train()
    cl.train()
    disc_l.train()
    disc_v.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()


    end = time.time()
    

    bar = Bar('Processing', max=len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)


        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        
        #update class
        '''
        imput_show = inputs[1,...]
        imput_show = imput_show[0,...]
        label_show = targets[1,...]

        print('mmmmm',inputs.shape,targets.shape)
        plt.figure()
        plt.imshow(imput_show.cpu())
        plt.show()
        '''
        u = np.random.uniform(-1, 1, (args.train_batch, 288, 1, 1))   
        l2 = torch.from_numpy(u).float().cuda()

        dec_l2 = dec(l2)
        n = torch.randn(args.train_batch, 1, 28, 28).cuda()
        l1 = enc(inputs + n)
        logits_C_l1 = cl(dec(l1))
        logits_C_l2 = cl(dec_l2)

        valid_logits_C_l1 = Variable(Tensor(logits_C_l1.shape[0], 1).fill_(1.0), requires_grad=False)
        fake_logits_C_l2 = Variable(Tensor(logits_C_l2.shape[0], 1).fill_(0.0), requires_grad=False)

        loss_cl_l1 = criterion_ce(logits_C_l1,valid_logits_C_l1)
        loss_cl_l2 = criterion_ce(logits_C_l2,fake_logits_C_l2)

        loss_cl = (loss_cl_l1 + loss_cl_l2 ) / 2

        cl.zero_grad()
        loss_cl.backward(retain_graph=True)
        optimizer_c.step()


        disc_l_l1 = l1.view(l1.size(0),32,3,3)
        disc_l.zero_grad()
        logits_Dl_l1 = disc_l(disc_l_l1)
        logits_Dl_l2 = disc_l(l2)
        dl_logits_DL_l1 = Variable(Tensor(logits_Dl_l1.shape[0], 1).fill_(0.0), requires_grad=False)
        dl_logits_DL_l2 = Variable(Tensor(logits_Dl_l2.shape[0], 1).fill_(1.0), requires_grad=False)
 
        loss_dl_1 = criterion_ce(logits_Dl_l1 , dl_logits_DL_l1)
        loss_dl_2 = criterion_ce(logits_Dl_l2 , dl_logits_DL_l2)
        loss_dl = (loss_dl_1 + loss_dl_2) / 2

        
        loss_dl.backward(retain_graph=True)
        optimizer_dl.step()


        logits_Dv_X = disc_v(inputs)
        logits_Dv_l2 = disc_v(dec(l2))


        dv_logits_Dv_X = Variable(Tensor(logits_Dv_X.shape[0], 1).fill_(1.0), requires_grad=False)
        dv_logits_Dv_l2 = Variable(Tensor(logits_Dv_l2.shape[0], 1).fill_(0.0), requires_grad=False)
        
        loss_dv_1 = criterion_ce(logits_Dv_X,dv_logits_Dv_X)
        loss_dv_2 = criterion_ce(logits_Dv_l2,dv_logits_Dv_l2)
        loss_dv = (loss_dv_1 + loss_dv_2) / 2

        disc_v.zero_grad()
        loss_dv.backward()
        optimizer_dv.step()



        for i in range(5):
            logits_C_l2_mine = cl(dec(l2))
            zeros_logits_C_l2_mine = Variable(Tensor(logits_C_l2_mine.shape[0], 1).fill_(0.0), requires_grad=False)
            loss_C_l2_mine = criterion_ce(logits_C_l2_mine,zeros_logits_C_l2_mine)
            optimizer_l2.zero_grad()
            loss_C_l2_mine.backward()
            optimizer_l2.step()

        ######  update ae 
        out_gv1 = disc_v(dec(l2))
        Xh = dec(l1)
        loss_mse = criterion_ae(Xh,inputs)
        ones_logits_Dl_l1 = Variable(Tensor(logits_Dl_l1.shape[0], 1).fill_(1.0), requires_grad=False)
 
        loss_AE_l = criterion_ce(logits_Dl_l1,ones_logits_Dl_l1)

        logits_Dv_l2_mine = disc_v(dec_l2)
        ones_logits_Dv_l2_mine = Variable(Tensor(logits_Dv_l2_mine.shape[0], 1).fill_(1.0), requires_grad=False)
        loss_ae_v = criterion_ce(logits_Dv_l2_mine,ones_logits_Dv_l2_mine)

        loss_ae_all = 10*loss_mse + loss_ae_v + loss_AE_l

        enc.zero_grad()
        dec.zero_grad()
        loss_ae_all.backward()
        optimizer_en.step()
        optimizer_de.step()

        losses.update(loss_ae_all.item(), inputs.size(0))



       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f}  '.format(
                    batch=batch_idx + 1,
                    size=len(trainloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,

                    )
        bar.next()
    bar.finish()
    
    #save images during training time
    if epoch % 5 == 0:
        recon = dec(enc(inputs))
        recon = recon.cpu().data
        inputs = inputs.cpu().data
        if not os.path.exists('./result/0000/train_dc_fake-1'):
            os.mkdir('./result/0000/train_dc_fake-1')
        if not os.path.exists('./result/0000/train_dc_real-1'):
            os.mkdir('./result/0000/train_dc_real-1')
        save_image(recon, './result/0000/train_dc_fake-1/fake_0{}.png'.format(epoch))
        save_image(inputs, './result/0000/train_dc_real-1/real_0{}.png'.format(epoch))  
    return losses.avg

def test(args,testloader, enc, dec,cl,disc_l,disc_v, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()


    # switch to evaluate mode
    enc.eval()
    dec.eval()
    cl.eval()
    disc_l.eval()
    disc_v.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))

    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # with torch.no_grad():
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
        # compute output
        recon = dec(enc(inputs))       
        scores = torch.mean(torch.pow((inputs - recon), 2),dim=[1,2,3])
        prec1 = roc_auc_score(targets.cpu().detach().numpy(), -scores.cpu().detach().numpy())

        top1.update(prec1, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | top1: {top1: .4f} '.format(
                    batch=batch_idx + 1,
                    size=len(testloader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    top1=top1.avg,
                    )
        bar.next()
    bar.finish()
    return top1.avg