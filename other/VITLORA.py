import shutil
from utils import *
from update import *
from tqdm import tqdm
from CPN import *
from datasets import load_dataset
import math
from utils import _load_clinc150_data, _load_fewrel_data, _load_trace_data, save_old_param, save_global_tau
import torch
from transformers import (
    MODEL_MAPPING,
    AdamW,
    get_scheduler,
)
import logging
from networks import fisher_model, ldbr_model
from networks.buffer import FixedSizeBuffer
from sklearn.metrics import f1_score, confusion_matrix
from FedCLModule import FedCLModule
from src.MABFedCL import MABFedCL
import os
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class vitlora:
    def __init__(self, args, task_size, device, accelerator):
        self.all_tasks_completed = None
        self.data_dir = '/home/qiuwenqi/LLM/Datasets/banking77'  # 假设数据集文件已经上传到此路径
        self.args = args
        self.numclass = args.fg_nc
        self.task_size = task_size
        self.device = device
        self.task_accuracies = []
        self.previous_task_accuracies = []
        self.list_of_individual_testloader = []
        self.classes = None
        self.old_model = None
        self.list_of_testloader = list()
        self.lora_M = None
        self.lora_Q = None


        # -----------------------
        self._load_datasets()
        update_args(self.args)
        self.global_model, self.data_collator, self.tokenizer = initialize_model(self.args)

        # -----------------------
        if 'AMAFCL' in self.args.baseline:

            if self.args.task != 0:
                last_preserve_path = os.path.join(self.args.last_ckpt, "preserved_server_state.pt")
                last_preserve_param = torch.load(last_preserve_path, map_location="cpu")
                self.prev_meta_params = last_preserve_param['meta_params']
                self.prev_similarity = last_preserve_param['similarity']
                self.prev_pold = last_preserve_param['P_old']
                self.prev_rold = last_preserve_param['R_old']
                self.fedcl_module = FedCLModule(
                    self.global_model,
                    encoder=None,
                    readout_fn=None,
                    args=self.args,
                    accelerator=accelerator,
                    lr=self.args.encoders_lr,
                    tau=0.6,
                    lambda_reg=0.1,
                    topk_ratio=self.args.topk_ratio,
                    prev_meta_params=self.prev_meta_params,
                    prev_similarity=self.prev_similarity,
                    prev_pold=self.prev_pold,
                    prev_rold=self.prev_rold
                )
            else:
                self.fedcl_module = FedCLModule(
                    self.global_model,
                    encoder=None,
                    readout_fn=None,
                    args=self.args,
                    accelerator=accelerator,
                    lr=self.args.encoders_lr,
                    tau=0.6,
                    lambda_reg=0.1,
                    topk_ratio=self.args.topk_ratio,
                    prev_meta_params=None,
                    prev_similarity=None,
                    prev_pold=None,
                    prev_rold=None
                )
        elif 'MABFedCL' in self.args.baseline:
            self.fedcl_module = MABFedCL(
                self.global_model,
                args=self.args,
                accelerator=accelerator,
                tau=0.6,
                topk_ratio=self.args.topk_ratio
            )
            # if self.args.task > 0:
            #     mab_path = os.path.join(self.args.last_ckpt, 'preserved_mab_state.pt')
            #     mab_state = torch.load(mab_path, map_location='cpu')
            #     tau = mab_state.get('tau', 0.6)
            #     topk_ratio = mab_state.get('topk_ratio', self.args.topk_ratio)
            # else:
            #     tau = 0.6
            #     topk_ratio = self.args.topk_ratio
            # # 初始化新的 MABFedCL 算法模块
            # self.fedcl_module = MABFedCL(
            #     self.global_model,
            #     args=self.args,
            #     accelerator=accelerator,
            #     tau=tau,
            #     topk_ratio=topk_ratio
            # )







    def _load_datasets(self):
        if "banking" in self.args.dataset:
            print("Using data from banking 77 dataset")
            dataset = load_dataset('csv',
                                   data_files={'train': f"{self.data_dir}/train.csv",
                                               'test': f"{self.data_dir}/test.csv"},
                                   delimiter=',')  # 确保读取CSV格式，指定分隔符
            # 重命名列
            # dataset = dataset.rename_column(original_column_name='text', new_column_name='text')
            dataset = dataset.rename_column(original_column_name='category', new_column_name='labels')

            # dataset = get_dataset("fewrel", tokenizer=None, args=self.args)

            # 分割训练集和验证集
            self.train_set = dataset['train']
            self.test_set = dataset['test']

        elif "clinc" in self.args.dataset:

            print("Using data from clinc150 dataset")
            # 加载并合并 clinc150 数据集
            clinc150_train, clinc150_test = _load_clinc150_data(
                clinc150_data_path='/home/qiuwenqi/LLM/Datasets/clinc150/data_full.json'
            )

            self.train_set = clinc150_train
            self.test_set = clinc150_test

        elif "fewrel" in self.args.dataset:
            print("Using data from FewRel dataset")

            # dataset = get_dataset("fewrel", tokenizer=None, args=self.args)

            # 加载 fewrel 数据集
            fewrel_train, fewrel_test = _load_fewrel_data(
                fewrel_data_path='/home/qiuwenqi/LLM/Datasets/FewRel/FewRel-2021.pkl'
            )

            self.train_set = fewrel_train
            self.test_set = fewrel_test

        elif "trace" in self.args.dataset:
            print("Using data from traced dataset")

            trace_train, trace_test = _load_trace_data(
                trace_data_path='/home/qiuwenqi/LLM/Datasets/trace/TACRED-2021.pkl'
            )
            # dataset_train = fewrel_train.rename_column(original_column_name='text', new_column_name='input_text')
            # dataset_test = fewrel_test.rename_column(original_column_name='labels', new_column_name='label')

            trace_train = trace_train.rename_column('text', 'input_text')
            trace_train = trace_train.rename_column('labels', 'label')

            trace_test = trace_test.rename_column('text', 'input_text')
            trace_test = trace_test.rename_column('labels', 'label')

            self.train_set = trace_train
            self.test_set = trace_test


        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

    def save_model(self, accelerator, model):
        """保存 MyBart 模型的状态字典和配置"""
        unwrapped_model = accelerator.unwrap_model(model)  # 解包分布式模型
        output_dir = self.args.output_dir

        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")

        # 保存 MyBart 模型的状态字典和 args 配置
        accelerator.save({
            "state_dict": unwrapped_model.state_dict(),  # 保存完整的模型权重
            "args": vars(self.args)  # 保存当前任务的配置参数
        }, f"{output_dir}/mybart_checkpoint.pt")

        print(f"MyBart model saved to {output_dir}/mybart_checkpoint.pt")

    def load_model(self, model):
        model_dict_path = os.path.join(self.args.output_dir, 'pytorch_model.bin')
        if 'l2p' in self.args.baseline:
            model.load_state_dict(torch.load(model_dict_path, map_location='cpu'))
        else:
            model.model.load_state_dict(torch.load(model_dict_path, map_location='cpu'))

    def raw_train(self, current_task, accelerator=None, dev_loader=None):
        """集中式增量式训练的实现"""

        self.train_loader = DataLoader(self.train_set, batch_size=self.args.local_bs, shuffle=True, num_workers=0,
                                       collate_fn=self.data_collator)

        if 'ewc' in self.args.baseline:
            if os.path.exists(os.path.join(self.args.prev_output, 'fisher')):
                print('load fisher matrix **************')
                self_fisher = torch.load(os.path.join(self.args.prev_output, 'fisher'))
                for k, v in self_fisher.items():
                    self_fisher[k] = self_fisher[k].cuda()
            else:
                self_fisher = None

        if 'experience_replay' in self.args.baseline or 'derpp' in self.args.baseline:
            # Load buffer.
            if self.args.task == 0:
                buffer = FixedSizeBuffer(buffer_size=self.args.store_ratio)
            else:
                buffer = torch.load(os.path.join(self.args.model_name_or_path, 'buffer.pth'))

        if 'ldbr' in self.args.baseline:
            predictor = ldbr_model.Predictor(2, hidden_size=128).cuda()
            buffer = ldbr_model.Memory()
            if self.args.task > 0:
                buffer.load(os.path.join(self.args.model_name_or_path, 'buffer.json'))
                predictor.load_state_dict(
                    torch.load(os.path.join(self.args.model_name_or_path, 'predictor.pth'), map_location='cpu'),
                )
                predictor = predictor.cuda()

            optimizer_P = AdamW(
                [
                    {"params": predictor.parameters(), "lr": self.args.classifier_lr, "weight_decay": 0.01},
                ]
            )
            optimizer_P = accelerator.prepare(optimizer_P)

        network_params = []
        if self.args.is_peft:
            for name, param in self.global_model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    network_params.append({'params': param, 'lr': self.args.encoders_lr})
        else:
            for param in self.global_model.parameters():
                network_params.append({'params': param, 'lr': self.args.encoders_lr})

        optimizer = AdamW(network_params)

        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epochs * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        if self.args.lr_scheduler_type == 'none':
            lr_scheduler = None
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.max_train_steps,
            )

        model, optimizer, train_loader = accelerator.prepare(self.global_model, optimizer, self.train_loader)

        if dev_loader is not None:
            dev_loader = accelerator.prepare(dev_loader)

        if 'ldbr' in self.args.baseline:
            buffer.store_features(model)
            currentBuffer = ldbr_model.Memory()
            model.eval()
            print("INIT current buffer...")
            with torch.no_grad():
                for inputs in train_loader:
                    for i in range(inputs['input_ids'].shape[0]):
                        currentBuffer.append(
                            inputs['input_ids'][i].data.cpu().numpy(),
                            inputs['attention_mask'][i].data.cpu().numpy(),
                            inputs['labels'][i].item(),
                            self.args.task
                        )
            print("Start Storing Features...")
            currentBuffer.store_features(model)
            length = len(currentBuffer)

        if accelerator.is_main_process:
            logger.info("***** Running training *****")
            logger.info(
                f"Pretrained Model = {self.args.model_name_or_path},  Dataset name = {self.args.dataset}, "
                f"seed = {self.args.seed}, test size = {len(self.test_set)}, training size = {len(self.train_set)}")
            logger.info(
                f"  Learning Rate = {self.args.encoders_lr}, Classifier Learning Rate = {self.args.classifier_lr},"
                f" Warmup Num = {self.args.num_warmup_steps}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f"  Seq ID = {self.args.idrandom}, Task id = {current_task}, dataset name = {self.args.dataset},"
                f" Num task = {self.args.total_num}")
            logger.info(
                f"  Baseline = {self.args.baseline}, Batch Size = {self.args.local_bs}, Epoch= {self.args.epochs}")

        global_step = 0

        if accelerator.is_main_process:
            # Delete previous models if we do not want to save all checkpoints.
            if 'save_all_ckpt' not in self.args.baseline:
                for saved_output_dir in self.args.saved_output_dir[:-2]:  # We need -2 so that we can load model.
                    if os.path.isdir(saved_output_dir):
                        shutil.rmtree(saved_output_dir)

        print(100 * '#')
        print("Begin Training!")

        # Train
        for epoch in range(self.args.epochs):

            total_loss = 0
            total_num = 0

            if 'ldbr' in self.args.baseline:
                iteration = 1
                progress_bar = tqdm(currentBuffer.get_minibatch(self.args.batch_size),
                                    total=length // self.args.batch_size, ncols=100,
                                    disable=not accelerator.is_local_main_process)

                for x, mask, y, t, origin_fea in progress_bar:

                    if iteration % 10 == 0 and self.args.task > 0:
                        # Replay.
                        total_x, total_mask, total_y, total_t, total_fea = x, mask, y, t, origin_fea
                        for j in range(self.args.task):
                            old_x, old_mask, old_y, old_t, old_fea = \
                                buffer.get_random_batch(self.args.batch_size, j)
                            total_x = torch.cat([old_x, total_x], dim=0)
                            total_mask = torch.cat([old_mask, total_mask], dim=0)
                            total_y = torch.cat([old_y, total_y], dim=0)
                            total_t = torch.cat([old_t, total_t], dim=0)
                            total_fea = torch.cat([old_fea, total_fea], dim=0)
                        for j in range(self.args.task + 1):
                            x = total_x[j * self.args.batch_size: (j + 1) * self.args.batch_size, :]
                            mask = total_mask[j * self.args.batch_size: (j + 1) * self.args.batch_size, :]
                            y = total_y[j * self.args.batch_size: (j + 1) * self.args.batch_size]
                            t = total_t[j * self.args.batch_size: (j + 1) * self.args.batch_size]
                            fea = total_fea[j * self.args.batch_size: (j + 1) * self.args.batch_size, :]
                            x, mask, y, t, fea = \
                                x.cuda(), mask.cuda(), y.cuda(), t.cuda(), fea.cuda()
                            loss = ldbr_model.train_step(model, x, mask, y, t, self.args.task, True, fea, predictor)
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer_P.step()
                            optimizer.zero_grad()
                            optimizer_P.zero_grad()

                        iteration += 1
                        global_step += 1
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        progress_bar.set_description('Train Iter (Epoch=%3d,loss=%5.3f)' % (
                            (epoch, loss.item())))  # show the loss

                    else:
                        x, mask, y, t, origin_fea = x.cuda(), mask.cuda(), y.cuda(), t.cuda(), origin_fea.cuda()
                        # if self.args.dataset_name == 'tacred':
                        #     import pdb
                        #     pdb.set_trace()
                        loss = \
                            ldbr_model.train_step(model, x, mask, y, t, self.args.task, False, origin_fea,
                                                  predictor)

                        iteration += 1
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer_P.step()

                        global_step += 1
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        optimizer.zero_grad()
                        optimizer_P.zero_grad()

                        progress_bar.set_description('Train Iter (Epoch=%3d,loss=%5.3f)' % (
                            (epoch, loss.item())))  # show the loss

            else:
                progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_local_main_process)
                # batch 改成了 inputs
                for batch_idx, inputs in enumerate(train_loader):
                    model.train()

                    if 'ewc' in self.args.baseline:
                        if 'bart_classification' in self.args.baseline:
                            outputs = model(**inputs, self_fisher=self_fisher)
                        else:
                            outputs = model(inputs, self_fisher=self_fisher)

                    elif 'l2p' in self.args.baseline:
                        outputs = model(**inputs)

                    elif 'experience_replay' in self.args.baseline or 'derpp' in self.args.baseline:
                        if 'bart' in self.args.baseline:
                            outputs = model(**inputs, buffer=buffer)
                        else:
                            outputs = model(inputs, buffer=buffer)

                    elif 'bart_classification' in self.args.baseline:
                        outputs = model(**inputs, restrict_label=True)

                    else:
                        outputs = model(inputs)

                    loss = outputs.loss

                    # 完全不一样了，基于lora的最后一层输出跟预训练model的输出差了好几个量级
                    if 'distill' in self.args.baseline:
                        distill_loss = outputs.distill_loss
                        loss = loss + self.args.lamb_distill * distill_loss

                    accelerator.backward(loss)

                    # if accelerator.is_main_process and epoch < 1 and batch_idx < 1:
                    #     for n, p in model.named_parameters():
                    #         if p.grad is not None:
                    #             print('n,p： ', n, p.size())

                    optimizer.step()

                    global_step += 1
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()

                    progress_bar.update(1)
                    progress_bar.set_description(
                        'Train Iter (Epoch=%3d,loss=%5.3f)' % (epoch, loss.item()))

                    total_loss += loss.data.cpu().numpy().item() * inputs['input_ids'].size(0)
                    total_num += inputs['input_ids'].size(0)

            if self.args.eval_every_epoch:
                # We track the current task performance in every epoch.
                test_loader = self.test_loader
                test_loader = accelerator.prepare(test_loader)
                micro_f1, macro_f1, acc, _, _, _, _, _, _, _, _, _ = self.eval(model, test_loader, accelerator)
                logger.info(
                    "Epoch {} macro_f1 = {:.4f}, acc = {:.4f}, average loss = {:.4f} (seed={})".format(
                        epoch, macro_f1, acc, total_loss / total_num, self.args.seed))

            if dev_loader is not None:
                micro_f1, macro_f1, acc, _, _, _, _, _, _, _, _, _ = self.eval(model, dev_loader, accelerator)
                logger.info(
                    "**Dev set performance** Epoch {} macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(
                        epoch, macro_f1, acc, self.args.seed))
                if acc <= best_dev_result:
                    # We use the dev set for early stopping. Load the best model on dev set and stop training.
                    self.load_model(model)
                    break
                else:
                    best_dev_result = acc
                    self.save_model(accelerator, model)
                if epoch == (self.args.epoch - 1):
                    self.save_model(accelerator, model)

        # After training ***********************************************************************************************
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if dev_loader is None:
                # If we don't use dev set for early stopping, we save the model after the training is finished.
                self.save_model(accelerator, model)
                # self.last_model = model

            # self.tokenizer.save_pretrained(self.args.output_dir)

            if 'ldbr' in self.args.baseline:
                torch.save(predictor.state_dict(), os.path.join(self.args.output_dir, 'predictor.pth'))
                print("select samples to store....")
                ldbr_model.select_samples_to_store(model, buffer, train_loader, self.args.task, self.args.store_ratio)
                buffer.save(os.path.join(self.args.output_dir, 'buffer.json'))

        if 'ewc' in self.args.baseline:
            fisher_model.fisher_compute(train_loader, model, self_fisher, accelerator, self.args)

        elif 'experience_replay' in self.args.baseline:
            # Make sure the random seeds are different when running different tasks. Otherwise, the reservoir sampling
            # is not truly random.
            np.random.seed(self.args.seed * train_loader.dataset['labels'][0].item())
            # Add new data to the buffer and save the new buffer.
            for _, inputs in enumerate(train_loader):
                buffer.add_data(inputs['input_ids'],
                                labels=inputs['labels'],
                                attention_mask=inputs['attention_mask'])
            print(f'The buffer now contains {buffer.num_seen_examples} examples!')
            torch.save(buffer, os.path.join(self.args.output_dir, 'buffer.pth'))

        elif 'derpp' in self.args.baseline:
            # We also need to save the logits.
            model.eval()
            with torch.no_grad():
                for _, inputs in enumerate(train_loader):
                    outputs = model(**inputs)
                    logits = outputs.logits.cpu()
                    buffer.add_data(inputs['input_ids'],
                                    labels=inputs['labels'],
                                    logits=logits,
                                    attention_mask=inputs['attention_mask'])
            print(f'The buffer now contains {buffer.num_seen_examples} examples!')
            torch.save(buffer, os.path.join(self.args.output_dir, 'buffer.pth'))

        total_correct_cnt = 0
        total_sample_cnt = 0
        total_til_correct_cnt = 0  # within-task prediction
        total_tid_correct_cnt = 0  # task-id prediction
        predictions = []
        labels = []

        # Evaluation
        for eval_t in range(current_task + 1):  # Test one all seen classes.
            self.args.task = eval_t

            test_loader = self.list_of_individual_testloader[eval_t]
            test_loader = accelerator.prepare(test_loader)
            micro_f1, macro_f1, acc, test_loss, correct_cnt, sample_cnt, pred_list, label_list, til_acc, \
                til_correct_cnt, tid_acc, tid_correct_cnt = \
                self.eval(model, test_loader, accelerator, self.task_masks[eval_t])
            total_sample_cnt += sample_cnt
            total_correct_cnt += correct_cnt
            total_til_correct_cnt += til_correct_cnt
            total_tid_correct_cnt += tid_correct_cnt
            predictions += pred_list
            labels += label_list

            if accelerator.is_main_process:

                logger.info(
                    "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(
                        self.args.model_name_or_path,
                        self.args.dataset, macro_f1,
                        acc, self.args.seed))

                progressive_f1_path = os.path.join(self.args.output_dir + '/../',
                                                   'progressive_f1_' + str(self.args.seed))
                progressive_acc_path = os.path.join(self.args.output_dir + '/../',
                                                    'progressive_acc_' + str(self.args.seed))
                progressive_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                                'accumulated_acc_' + str(self.args.seed))
                print('progressive_f1_path: ', progressive_f1_path)
                print('progressive_acc_path: ', progressive_acc_path)
                print('progressive_accumulated_acc_path: ', progressive_accumulated_acc_path)

                # Calculate the TIL results and task-id prediction results for analysis.
                progressive_til_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'til_progressive_acc_' + str(self.args.seed))
                til_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'til_accumulated_acc_' + str(self.args.seed))
                progressive_tid_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'tid_progressive_acc_' + str(self.args.seed))
                tid_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'tid_accumulated_acc_' + str(self.args.seed))

                if os.path.exists(progressive_f1_path) and os.path.exists(progressive_acc_path):
                    f1s = np.loadtxt(progressive_f1_path)
                    accs = np.loadtxt(progressive_acc_path)
                else:
                    f1s = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)
                    accs = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)

                if os.path.exists(progressive_accumulated_acc_path):
                    accumulated_accs = np.loadtxt(progressive_accumulated_acc_path)
                else:
                    accumulated_accs = np.zeros(self.args.total_num, dtype=np.float32)

                if os.path.exists(progressive_til_acc_path) and os.path.exists(progressive_tid_acc_path):
                    til_accs = np.loadtxt(progressive_til_acc_path)
                    tid_accs = np.loadtxt(progressive_tid_acc_path)
                else:
                    til_accs = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)
                    tid_accs = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)

                if os.path.exists(til_accumulated_acc_path) and os.path.exists(tid_accumulated_acc_path):
                    til_accumulated_accs = np.loadtxt(til_accumulated_acc_path)
                    tid_accumulated_accs = np.loadtxt(tid_accumulated_acc_path)
                else:
                    til_accumulated_accs = np.zeros(self.args.total_num, dtype=np.float32)
                    tid_accumulated_accs = np.zeros(self.args.total_num, dtype=np.float32)

                f1s[current_task][eval_t] = macro_f1
                np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

                accs[current_task][eval_t] = acc
                np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

                til_accs[current_task][eval_t] = til_acc
                np.savetxt(progressive_til_acc_path, til_accs, '%.4f', delimiter='\t')

                tid_accs[current_task][eval_t] = tid_acc
                np.savetxt(progressive_tid_acc_path, tid_accs, '%.4f', delimiter='\t')

                if eval_t == current_task:  # Test results on all available test data.
                    accumulated_accs[eval_t] = total_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(progressive_accumulated_acc_path, accumulated_accs, '%.4f', delimiter='\t')
                    til_accumulated_accs[eval_t] = total_til_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(til_accumulated_acc_path, til_accumulated_accs, '%.4f', delimiter='\t')
                    tid_accumulated_accs[eval_t] = total_tid_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(tid_accumulated_acc_path, tid_accumulated_accs, '%.4f', delimiter='\t')

                if current_task == self.args.total_num - 1:  # last ft task, we need a final one
                    final_f1 = os.path.join(self.args.output_dir + '/../', 'f1_' + str(self.args.seed))
                    final_acc = os.path.join(self.args.output_dir + '/../', 'acc_' + str(self.args.seed))

                    forward_f1 = os.path.join(self.args.output_dir + '/../', 'forward_f1_' + str(self.args.seed))
                    forward_acc = os.path.join(self.args.output_dir + '/../', 'forward_acc_' + str(self.args.seed))

                    print('final_f1: ', final_f1)
                    print('final_acc: ', final_acc)

                    # Save the confusion matrix.
                    cm = confusion_matrix(y_true=labels, y_pred=predictions, normalize='true')
                    np.savetxt(self.args.output_dir + '/../confusion_matrix', cm, '%.4f', delimiter='\t')

                    if self.args.baseline == 'one':
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')

                    else:
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[-1][j]) + '\n')
                                f1_file.writelines(str(f1s[-1][j]) + '\n')

                        with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')
        # Save the training arguments.
        training_args = {k: v for k, v in self.args.__dict__.items() if k != 'device'}
        dump_json(training_args, self.args.output_dir + '/../training_args.json')

    def setup_data(self, shuffle):
        # 获取训练集和测试集的类别标签
        train_targets = self.train_set['labels']
        test_targets = self.test_set['labels']
        unique_classes = sorted(set(train_targets) | set(test_targets))
        self.total_classes = len(unique_classes)
        root_dir = os.path.join(
            self.args.base_dir,
            f"seq_{self.args.idrandom}_seed{self.args.seed}",
            str(self.args.baseline),
            str(self.args.dataset),
            f"topK_{self.args.topk_ratio}"
        )
        os.makedirs(root_dir, exist_ok=True)
        class_order_path = os.path.join(root_dir, 'class_order.json')

        if self.args.task == 0:
            self.class_order = np.random.permutation(unique_classes).tolist() if shuffle else unique_classes
            dump_json(self.class_order, class_order_path)
            print(f"[Task 0] Saved class order to {class_order_path}")
        else:
            if os.path.exists(class_order_path):
                self.class_order = load_json(class_order_path)  # 注意：不再 map(int, ...)
                print(f"[Task {self.args.task}] Loaded class order from {class_order_path}")
            else:
                raise FileNotFoundError(f"class_order.json not found at {class_order_path}. "
                                        f"Please make sure task 0 has run and class order is saved.")

        label_mapping = {old_label: new_label for new_label, old_label in enumerate(self.class_order)}

        # 使用 map() 更新数据集标签
        train_set_m = self.train_set.map(lambda example: {'labels': label_mapping[example['labels']]})
        test_set_m = self.test_set.map(lambda example: {'labels': label_mapping[example['labels']]})

        def preprocess_function(examples):
            return self.tokenizer(
                examples['text'],
                padding='max_length',
                truncation=True,
                max_length=self.args.max_seq_length
            )

        # 对整个测试集进行一次性预处理
        test_set = test_set_m.map(preprocess_function, batched=True)
        test_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        train_set = train_set_m.map(preprocess_function, batched=True)
        train_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        current_task = self.args.task

        # 仅计算当前任务及以前任务的类别范围
        self.classes_cache = {}
        for task in range(current_task + 1):
            if task == 0:
                start_class = 0
                end_class = self.args.fg_nc
            else:
                start_class = self.args.fg_nc + (task - 1) * self.task_size
                end_class = min(self.args.fg_nc + task * self.task_size, self.total_classes)
            self.classes_cache[task] = [start_class, end_class]

        print("Preprocessing data for tasks up to", current_task)
        self.task_train_sets = {}
        self.current_test_set = {}
        self.task_masks = {}

        for task in range(current_task + 1):
            start_class, end_class = self.classes_cache[task]

            if task == current_task:
                self.task_train_sets[task] = train_set.filter(
                    lambda example: start_class <= example['labels'] < end_class
                )

            self.current_test_set[task] = test_set.filter(
                    lambda example: start_class <= example['labels'] < end_class
            )

            task_mask = torch.zeros(80)
            for idx in range(start_class, end_class):
                task_mask[idx] = 1
            self.task_masks[task] = task_mask

    def beforeTrain(self, current_task):

        self.classes, self.all_tasks_completed = before_train_utils(current_task,
                                                                    self.classes_cache,
                                                                    self.total_classes)
        if self.all_tasks_completed:
            return

        print(f"Now is training task {current_task}")
        print(f"train_class is {self.classes[0]} to {self.classes[1]}")
        print(f"test_class is {self.classes[0]} to {self.classes[1]}")

        self.train_set = self.task_train_sets[current_task]

        for i in range(self.args.task + 1):
            self.current_test = self.current_test_set[i]
            individual_test_loader = DataLoader(
                self.current_test, batch_size=self.args.local_bs, shuffle=False,
                num_workers=0, collate_fn=self.data_collator
            )
            self.list_of_individual_testloader.append(individual_test_loader)

    def train(self, current_task, accelerator, dev_loader):

        encoder_lr = self.args.encoders_lr
        # TODO 不是output_dir，要是一个固定的不随任务改变的路径
        # Load historical LoRA information
        if 'MABFedCL' in self.args.baseline:
            history_dir = os.path.abspath(os.path.join(self.args.output_dir, '..'))
            os.makedirs(history_dir, exist_ok=True)
            m_path = os.path.join(history_dir, 'lora_M.pt')
            q_path = os.path.join(history_dir, 'lora_Q.pt')
            if os.path.exists(m_path):
                self.lora_M = torch.load(m_path, map_location='cpu')['M']
            if os.path.exists(q_path):
                self.lora_Q = torch.load(q_path, map_location='cpu')['Q']

            # Record initial LoRA parameters for this task
            self._lora_start = {
                n: p.detach().clone() for n, p in self.global_model.named_parameters() if 'lora_A' in n
            }


        for epoch in tqdm(range(self.args.epochs)):
            sample_num = []
            m = self.args.client_local
            idxs_users = np.random.choice(range(self.args.num_users), m, replace=False).tolist()

            # load dataset and user groups
            train_dataset, user_groups = get_dataset_noniid(self.args, train_dataset=self.train_set,
                                                            m=m,
                                                            start=self.classes[0], end=self.classes[1],
                                                            task_num=self.task_size, idxs_users=idxs_users)
            grad_dist = {}

            client_prototypes = {}
            client_readouts = {}
            client_sample_counts = {}
            client_phi = {}
            client_tau = {}  # 每个客户端搜索得到的最佳 tau

            tau_global = None

            if 'MABFedCL' in self.args.baseline:
                # -------- Stage 1: clients search local tau ---------
                for idx in idxs_users:
                    local_data_indices = user_groups[idx]
                    sample_num.append(len(user_groups[idx]))
                    client_sample_counts[idx] = len(user_groups[idx])

                    client_dataset = Subset(train_dataset, local_data_indices)
                    train_loader = DataLoader(
                        client_dataset,
                        batch_size=self.args.local_bs,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=self.data_collator,
                    )

                    local_model_copy = copy.deepcopy(self.global_model)

                    mask = self.task_masks[current_task]  # 你已经构建好的 0/1 向量
                    local_model_copy.set_masked_label(mask)
                    # local_model_copy.masked_label = mask.to(accelerator.device)

                    _, _, _, _, _, tau_val = self.update_weights_local(
                        model=local_model_copy,
                        lr=encoder_lr,
                        train_loader=train_loader,
                        accelerator=accelerator,
                        dev_loader=None,
                        idx=idx,
                        current_task=current_task,
                        search_only=True,
                        global_tau=None,
                        Q=self.lora_Q,
                    )
                    client_tau[idx] = tau_val

                tau_global = self.fedcl_module.aggregate_tau(client_tau, client_sample_counts)

                # test

                save_global_tau(tau_global, current_task, epoch, self.args.output_dir)

                if accelerator.is_main_process:
                    logger.info(f"tau_global is {tau_global}, and client_tau {client_tau}")
            else:
                for idx in idxs_users:
                    # 每个客户端使用其对应的索引
                    local_data_indices = user_groups[idx]
                    # 每个节点样本数
                    sample_num.append(len(user_groups[idx]))
                    client_sample_counts[idx] = len(user_groups[idx])


            for idx in idxs_users:
                local_data_indices = user_groups[idx]
                client_dataset = Subset(train_dataset, local_data_indices)
                train_loader = DataLoader(client_dataset, batch_size=self.args.local_bs, shuffle=True,
                                          num_workers=0,
                                          collate_fn=self.data_collator)

                local_model_copy = copy.deepcopy(self.global_model)
                mask = self.task_masks[current_task]  # 你已经构建好的 0/1 向量
                local_model_copy.set_masked_label(mask)

                # local_model_copy.set_masked_label(self.task_masks[current_task])

                if 'AMAFCL' in self.args.baseline:
                    prototypes, R_i, total_samples, delta_model, phi_avg = self.update_weights_local(model=local_model_copy,
                                                                                            lr=encoder_lr,
                                                                                            train_loader=train_loader,
                                                                                            accelerator=accelerator,
                                                                                            dev_loader=None, idx=idx,
                                                                                            current_task=current_task)
                    grad_dist[idx] = delta_model
                    client_prototypes[idx] = prototypes
                    client_readouts[idx] = R_i
                    client_sample_counts[idx] = total_samples
                    client_phi[idx] = phi_avg


                elif 'olora' in self.args.baseline:
                    local_model, _ = self.update_weights_local(
                        model=local_model_copy,
                        lr=encoder_lr,
                        train_loader=train_loader,
                        accelerator=accelerator,
                        dev_loader=None,
                        idx=idx,
                        current_task=current_task,
                        Q=None  # 如果你还没引入 TopK，那可以传 None
                    )
                    grad, param = self.get_grad(local_model)
                    grad_dist[idx] = grad


                elif 'MABFedCL' in self.args.baseline:
                    prototypes, R_i, total_samples, delta_model, phi_avg, _ = self.update_weights_local(
                        model=local_model_copy,
                        lr=encoder_lr,
                        train_loader=train_loader,
                        accelerator=accelerator,
                        dev_loader=None, idx=idx,
                        current_task=current_task,
                        search_only=False,
                        global_tau=tau_global,
                        Q=self.lora_Q
                    )
                    grad_dist[idx] = delta_model
                    client_prototypes[idx] = prototypes
                    grad_dist[idx] = delta_model
                    client_prototypes[idx] = prototypes
                    client_readouts[idx] = R_i
                    client_sample_counts[idx] = total_samples
                    client_phi[idx] = phi_avg
                    # client_tau[idx] = tau_val

                else:
                    local_model, _ = self.update_weights_local(model=local_model_copy, lr=encoder_lr,
                                                               train_loader=train_loader, accelerator=accelerator,
                                                               dev_loader=None,
                                                               idx=idx, current_task=current_task,
                                                               Q=None)

                    grad, param = self.get_grad(local_model)
                    grad_dist[idx] = grad



            if 'AMAFCL' in self.args.baseline:
                tau, P_old_new, R_old_new, self.global_model, S_global, S_readout = self.fedcl_module.server_aggregate_and_meta_update(
                    client_updates=grad_dist,
                    client_prototypes=client_prototypes,
                    client_readouts=client_readouts,
                    client_sample_counts=client_sample_counts,
                    client_phi=client_phi,
                    lambda_KD=0.1,
                    lambda_conf=0.1
                )

            elif 'MABFedCL' in self.args.baseline:
                # 聚合阶段会同步各客户端的 tau
                _, _, _, self.global_model = self.fedcl_module.server_aggregate(
                    client_updates=grad_dist,
                    client_prototypes=client_prototypes,
                    client_readouts=client_readouts,
                    client_sample_counts=client_sample_counts,
                    client_tau=client_tau
                )

            else:
                grad, self.global_model = self.aggregate(grad_dist=grad_dist, cohorts=idxs_users,
                                                         partition_map=user_groups)

        accelerator.wait_for_everyone()

        if "olora" in self.args.baseline and self.args.is_peft:
            for name, param in self.global_model.named_parameters():
                if "lora_A" in name:
                    for name_, param_ in self.global_model.named_parameters():
                        if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                            new_A = torch.cat([param.data, param_.data], dim=0)
                            param.data = new_A.clone()
                elif "lora_B" in name:
                    for name_, param_ in self.global_model.named_parameters():
                        if "loranew_B" in name_ and name.split("lora_B")[0] == name_.split("loranew_B")[0]:
                            new_B = torch.cat([param.data, param_.data], dim=1)
                            param.data = new_B.clone()

        if accelerator.is_main_process:
            if dev_loader is None:
                # If we don't use dev set for early stopping, we save the model after the training is finished.
                self.save_model(accelerator, self.global_model)
                # self.last_model = self.global_model
            if 'AMAFCL' in self.args.baseline:
                save_old_param(tau, P_old_new, R_old_new, S_global, S_readout,
                               accelerator, self.args.output_dir)
            # if 'MABFedCL' in self.args.baseline:
            #     k_glob = getattr(self.fedcl_module, 'topk_ratio', self.args.topk_ratio)
            #     save_mab_state(tau, k_glob, accelerator, self.args.output_dir)
            logger.info(f"Preserved server state saved to {self.args.output_dir}")




        #----------------- Evaluation ----------------
        total_correct_cnt = 0
        total_sample_cnt = 0
        total_til_correct_cnt = 0  # within-task prediction
        total_tid_correct_cnt = 0  # task-id prediction
        predictions = []
        labels = []

        self.global_model = accelerator.prepare(self.global_model)

        # Evaluation
        for eval_t in range(current_task + 1):  # Test one all seen classes.
            self.args.task = eval_t

            test_loader = self.list_of_individual_testloader[eval_t]
            test_loader = accelerator.prepare(test_loader)
            # self.global_model = accelerator.prepare(self.global_model)
            micro_f1, macro_f1, acc, test_loss, correct_cnt, sample_cnt, pred_list, label_list, til_acc, \
                til_correct_cnt, tid_acc, tid_correct_cnt = \
                self.eval(self.global_model, test_loader, accelerator, self.task_masks[eval_t])
            total_sample_cnt += sample_cnt
            total_correct_cnt += correct_cnt
            total_til_correct_cnt += til_correct_cnt
            total_tid_correct_cnt += tid_correct_cnt
            predictions += pred_list
            labels += label_list

            if accelerator.is_main_process:

                logger.info(
                    "{} On {}, last epoch macro_f1 = {:.4f}, acc = {:.4f} (seed={})".format(
                        self.args.model_name_or_path,
                        self.args.dataset, macro_f1,
                        acc, self.args.seed))

                progressive_f1_path = os.path.join(self.args.output_dir + '/../',
                                                   'progressive_f1_' + str(self.args.seed))
                progressive_acc_path = os.path.join(self.args.output_dir + '/../',
                                                    'progressive_acc_' + str(self.args.seed))
                progressive_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                                'accumulated_acc_' + str(self.args.seed))
                print('progressive_f1_path: ', progressive_f1_path)
                print('progressive_acc_path: ', progressive_acc_path)
                print('progressive_accumulated_acc_path: ', progressive_accumulated_acc_path)

                # Calculate the TIL results and task-id prediction results for analysis.
                progressive_til_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'til_progressive_acc_' + str(self.args.seed))
                til_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'til_accumulated_acc_' + str(self.args.seed))
                progressive_tid_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'tid_progressive_acc_' + str(self.args.seed))
                tid_accumulated_acc_path = os.path.join(self.args.output_dir + '/../',
                                                        'tid_accumulated_acc_' + str(self.args.seed))

                if os.path.exists(progressive_f1_path) and os.path.exists(progressive_acc_path):
                    f1s = np.loadtxt(progressive_f1_path)
                    accs = np.loadtxt(progressive_acc_path)
                else:
                    f1s = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)
                    accs = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)

                if os.path.exists(progressive_accumulated_acc_path):
                    accumulated_accs = np.loadtxt(progressive_accumulated_acc_path)
                else:
                    accumulated_accs = np.zeros(self.args.total_num, dtype=np.float32)

                if os.path.exists(progressive_til_acc_path) and os.path.exists(progressive_tid_acc_path):
                    til_accs = np.loadtxt(progressive_til_acc_path)
                    tid_accs = np.loadtxt(progressive_tid_acc_path)
                else:
                    til_accs = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)
                    tid_accs = np.zeros((self.args.total_num, self.args.total_num), dtype=np.float32)

                if os.path.exists(til_accumulated_acc_path) and os.path.exists(tid_accumulated_acc_path):
                    til_accumulated_accs = np.loadtxt(til_accumulated_acc_path)
                    tid_accumulated_accs = np.loadtxt(tid_accumulated_acc_path)
                else:
                    til_accumulated_accs = np.zeros(self.args.total_num, dtype=np.float32)
                    tid_accumulated_accs = np.zeros(self.args.total_num, dtype=np.float32)

                f1s[current_task][eval_t] = macro_f1
                np.savetxt(progressive_f1_path, f1s, '%.4f', delimiter='\t')

                accs[current_task][eval_t] = acc
                np.savetxt(progressive_acc_path, accs, '%.4f', delimiter='\t')

                til_accs[current_task][eval_t] = til_acc
                np.savetxt(progressive_til_acc_path, til_accs, '%.4f', delimiter='\t')

                tid_accs[current_task][eval_t] = tid_acc
                np.savetxt(progressive_tid_acc_path, tid_accs, '%.4f', delimiter='\t')

                if eval_t == current_task:  # Test results on all available test data.
                    accumulated_accs[eval_t] = total_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(progressive_accumulated_acc_path, accumulated_accs, '%.4f', delimiter='\t')
                    til_accumulated_accs[eval_t] = total_til_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(til_accumulated_acc_path, til_accumulated_accs, '%.4f', delimiter='\t')
                    tid_accumulated_accs[eval_t] = total_tid_correct_cnt * 1.0 / total_sample_cnt
                    np.savetxt(tid_accumulated_acc_path, tid_accumulated_accs, '%.4f', delimiter='\t')

                if current_task == self.args.total_num - 1:  # last ft task, we need a final one
                    final_f1 = os.path.join(self.args.output_dir + '/../', 'f1_' + str(self.args.seed))
                    final_acc = os.path.join(self.args.output_dir + '/../', 'acc_' + str(self.args.seed))

                    forward_f1 = os.path.join(self.args.output_dir + '/../', 'forward_f1_' + str(self.args.seed))
                    forward_acc = os.path.join(self.args.output_dir + '/../', 'forward_acc_' + str(self.args.seed))

                    print('final_f1: ', final_f1)
                    print('final_acc: ', final_acc)

                    # Save the confusion matrix.
                    cm = confusion_matrix(y_true=labels, y_pred=predictions, normalize='true')
                    np.savetxt(self.args.output_dir + '/../confusion_matrix', cm, '%.4f', delimiter='\t')

                    if self.args.baseline == 'one':
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')

                    else:
                        with open(final_acc, 'w') as file, open(final_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[-1][j]) + '\n')
                                f1_file.writelines(str(f1s[-1][j]) + '\n')

                        with open(forward_acc, 'w') as file, open(forward_f1, 'w') as f1_file:
                            for j in range(accs.shape[1]):
                                file.writelines(str(accs[j][j]) + '\n')
                                f1_file.writelines(str(f1s[j][j]) + '\n')

        self.global_model = accelerator.unwrap_model(self.global_model)
        training_args = {k: v for k, v in self.args.__dict__.items() if k != 'device'}
        dump_json(training_args, self.args.output_dir + '/../training_args.json')

        if 'MABFedCL' in self.args.baseline:
            # After finishing the task, update LoRA history
            end_state = {n: p.detach().cpu() for n, p in self.global_model.named_parameters() if 'lora_A' in n}
            delta_vec = torch.cat([(end_state[n] - self._lora_start[n].cpu()).view(-1) for n in end_state])
            if self.lora_M is None:
                self.lora_M = delta_vec.unsqueeze(1)
            else:
                self.lora_M = torch.cat([self.lora_M, delta_vec.unsqueeze(1)], dim=1)
            k = min(self.args.orthogonal_k, self.lora_M.shape[1])
            try:
                U, S, V = torch.svd_lowrank(self.lora_M, q=k)
                self.lora_Q = U[:, :k]
            except Exception as e:
                logger.warning(f'SVD failed: {e}')
            torch.save({'M': self.lora_M.cpu()}, m_path)
            if self.lora_Q is not None:
                torch.save({'Q': self.lora_Q.cpu()}, q_path)




    def update_weights_local(self, model, lr, train_loader, accelerator, dev_loader, idx, current_task,
                             search_only=False, global_tau=None, Q=None):
        model.train()
        if accelerator.is_main_process:
            logger.info(f"Client {idx} Task {current_task}: 开始训练")

        client_dir = os.path.join(
            self.args.base_dir,
            f"seq_{self.args.idrandom}_seed{self.args.seed}",
            str(self.args.baseline),
            str(self.args.dataset),
            f"topK_{str(self.args.topk_ratio)}",
            f"client_idx_{idx}",  # 客户端标识
        )
        os.makedirs(client_dir, exist_ok=True)

        # 构建当前任务的输出目录
        current_output_dir = os.path.join(client_dir, f"task_{current_task}_model")
        os.makedirs(current_output_dir, exist_ok=True)

        # 定义 last_task 文件路径
        last_task_path = os.path.join(client_dir, 'last_task.txt')

        # 读取 last_task
        if os.path.exists(last_task_path):
            with open(last_task_path, 'r') as f:
                last_task_str = f.read().strip()
                last_task = int(last_task_str) if last_task_str.isdigit() else None
        else:
            last_task = None

        # 根据当前任务判断是否加载上一次的历史梯度
        if current_task == 0:
            loaded_hist_grad = None
        else:
            # 只有上一次任务编号存在且小于当前任务时才加载，否则视为未参与训练
            if last_task is not None and last_task < current_task:
                last_output_dir = os.path.join(client_dir, f"task_{last_task}_model")
                output_file_path = os.path.join(last_output_dir, 'historical_avg_grad.pt')
                if os.path.exists(output_file_path):
                    loaded = torch.load(output_file_path, map_location="cpu")
                    loaded_hist_grad = loaded.get("historical_avg_grad", None)
                    if loaded_hist_grad is not None:
                        loaded_hist_grad = loaded_hist_grad.detach()
                else:
                    loaded_hist_grad = None
            else:
                loaded_hist_grad = None

        if 'AMAFCL' in self.args.baseline:
            if accelerator.is_main_process:
                logger.info(f"Client {idx} Task {current_task}: 使用 AMAFCL 算法进行本地更新")

            # 构造优化器（仅优化 LoRA 层参数，当 is_peft 为 True）
            network_params = []
            if self.args.is_peft:
                for name, param in model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        network_params.append({'params': param, 'lr': lr})
            else:
                for param in model.parameters():
                    network_params.append({'params': param, 'lr': lr})
            from transformers import AdamW
            optimizer = AdamW(network_params)

            num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
            if self.args.max_train_steps is None:
                self.args.max_train_steps = self.args.local_ep * num_update_steps_per_epoch
            else:
                self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

            if self.args.lr_scheduler_type == 'none':
                lr_scheduler = None
            else:
                lr_scheduler = get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.num_warmup_steps,
                    num_training_steps=self.args.max_train_steps,
                )

            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

            # 调用新的接口 local_training_with_optimizer，直接传入已经包装好的 optimizer
            prototypes, R_i, total_samples, delta_model, phi_avg, _= self.fedcl_module.local_training(model, train_loader,
                                                                                           optimizer, lr_scheduler,
                                                                                           idx, current_output_dir,
                                                                                           historical_grad=loaded_hist_grad,
                                                                                           local_ep=self.args.local_ep,
                                                                                           current_task=current_task,
                                                                                            Q=self.lora_Q)



            if accelerator.is_main_process:
                logger.info(f"Client {idx} Task {current_task}: AMAFCL 本地训练结束")
            return prototypes, R_i, total_samples, delta_model, phi_avg

        elif 'MABFedCL' in self.args.baseline:
            if accelerator.is_main_process:
                logger.info(f"Client {idx} Task {current_task}: 使用 MABFedCL 算法进行本地更新")

            network_params = []
            if self.args.is_peft:
                for name, param in model.named_parameters():
                    if 'lora' in name.lower() and param.requires_grad:
                        network_params.append({'params': param, 'lr': lr})
            else:
                for param in model.parameters():
                    network_params.append({'params': param, 'lr': lr})
            from transformers import AdamW
            optimizer = AdamW(network_params)

            num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
            if self.args.max_train_steps is None:
                self.args.max_train_steps = self.args.local_ep * num_update_steps_per_epoch
            else:
                self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

            if self.args.lr_scheduler_type == 'none':
                lr_scheduler = None
            else:
                lr_scheduler = get_scheduler(
                    name=self.args.lr_scheduler_type,
                    optimizer=optimizer,
                    num_warmup_steps=self.args.num_warmup_steps,
                    num_training_steps=self.args.max_train_steps,
                )

            model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

            prototypes, R_i, total_samples, delta_model, phi_avg, tau_val = self.fedcl_module.local_training(
                model,
                train_loader,
                optimizer,
                lr_scheduler,
                idx,
                current_output_dir,
                historical_grad=loaded_hist_grad,
                local_ep=self.args.local_ep,
                current_task=current_task,
                global_tau=global_tau,
                search_only=search_only,
                Q=self.lora_Q,
            )
            # tau_val 为本轮搜索得到的最佳阈值

            if accelerator.is_main_process:
                logger.info(f"Client {idx} Task {current_task}: MABFedCL 本地训练结束")
            return prototypes, R_i, total_samples, delta_model, phi_avg, tau_val


        # EWC 相关
        if 'ewc' in self.args.baseline:
            if last_task is not None:
                # 构建上一个任务的输出路径
                last_output_dir = os.path.join(client_dir, f"task_{last_task}_model")
                fisher_path = os.path.join(last_output_dir, 'fisher')
                if os.path.exists(fisher_path):
                    print(f'Client {idx} Task {current_task}: 加载 Fisher 矩阵 **************')
                    try:
                        self_fisher = torch.load(fisher_path)
                        # for k, v in self_fisher.items():
                        #     self_fisher[k] = self_fisher[k].to(self.args.device)
                    except Exception as e:
                        print(f"加载 Fisher 矩阵失败: {e}")
                        self_fisher = None
                else:
                    print(f'Client {idx} Task {current_task}: 上一个任务 Fisher 矩阵不存在，跳过 EWC')
                    self_fisher = None
            else:
                print(f'Client {idx} Task {current_task}: 没有上一个任务，跳过 EWC')
                self_fisher = None

        if 'experience_replay' in self.args.baseline or 'derpp' in self.args.baseline:
            if current_task == 0:
                buffer = FixedSizeBuffer(buffer_size=self.args.store_ratio)
            else:
                buffer_path = os.path.join(current_output_dir, 'buffer.pth')
                buffer = torch.load(buffer_path) if os.path.exists(buffer_path) else FixedSizeBuffer(
                    buffer_size=self.args.store_ratio)

        if 'ldbr' in self.args.baseline:
            predictor = ldbr_model.Predictor(2, hidden_size=128).to(self.args.device)
            buffer = ldbr_model.Memory()
            if current_task > 0:
                buffer_path = os.path.join(current_output_dir, 'buffer.json')
                predictor_path = os.path.join(current_output_dir, 'predictor.pth')
                if os.path.exists(buffer_path):
                    buffer.load(buffer_path)
                if os.path.exists(predictor_path):
                    predictor.load_state_dict(torch.load(predictor_path, map_location='cpu'))
                predictor = predictor.to(self.args.device)
            from transformers import AdamW
            optimizer_P = AdamW(
                [
                    {"params": predictor.parameters(), "lr": self.args.classifier_lr, "weight_decay": 0.01},
                ]
            )
            optimizer_P = accelerator.prepare(optimizer_P)

        network_params = []
        if self.args.is_peft and 'olora' in self.args.baseline:
            for name, param in model.named_parameters():
                if 'loranew' in name.lower() and param.requires_grad:
                    network_params.append({'params': param, 'lr': lr})
        elif self.args.is_peft and 'lora' in self.args.baseline:
            for name, param in model.named_parameters():
                if 'lora' in name.lower() and param.requires_grad:
                    network_params.append({'params': param, 'lr': lr})
        else:
            for param in model.parameters():
                network_params.append({'params': param, 'lr': lr})
        from transformers import AdamW
        optimizer = AdamW(network_params)

        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.local_ep * num_update_steps_per_epoch
        else:
            self.args.epoch = math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)

        if self.args.lr_scheduler_type == 'none':
            lr_scheduler = None
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=self.args.num_warmup_steps,
                num_training_steps=self.args.max_train_steps,
            )

        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

        if 'ewc' in self.args.baseline:
            if self_fisher is not None:
                model_device = next(model.parameters()).device
                for key, value in self_fisher.items():
                    if isinstance(value, torch.Tensor):
                        self_fisher[key] = value.to(model_device)

        if dev_loader is not None:
            dev_loader = accelerator.prepare(dev_loader)

        if 'ldbr' in self.args.baseline:
            buffer.store_features(model)
            currentBuffer = ldbr_model.Memory()
            model.eval()
            print("INIT current buffer...")
            with torch.no_grad():
                for inputs in train_loader:
                    for i in range(inputs['input_ids'].shape[0]):
                        currentBuffer.append(
                            inputs['input_ids'][i].data.cpu().numpy(),
                            inputs['attention_mask'][i].data.cpu().numpy(),
                            inputs['labels'][i].item(),
                            self.args.task
                        )
            print("Start Storing Features...")
            currentBuffer.store_features(model)
            length = len(currentBuffer)

        if accelerator.is_main_process:
            logger.info("***** Running training in Local Client *****")
            logger.info(
                f"Client idx = {idx},  training size = {train_loader.total_dataset_length}")
            logger.info(
                f" Learning Rate = {self.args.encoders_lr}, Classifier Learning Rate = {self.args.classifier_lr},"
                f" Warmup Num = {self.args.num_warmup_steps}, Pre-trained Model = {self.args.model_name_or_path}")
            logger.info(
                f" Batch Size = {self.args.local_bs}, Local Epoch = {self.args.local_ep}")

        global_step = 0

        if accelerator.is_main_process:
            # Delete previous models if we do not want to save all checkpoints.
            if 'save_all_ckpt' not in self.args.baseline:
                for saved_output_dir in self.args.saved_output_dir[:-2]:  # We need -2 so that we can load model.
                    if os.path.isdir(saved_output_dir):
                        shutil.rmtree(saved_output_dir)
        if accelerator.is_main_process:
            print(100 * '#')
            print("Begin Local Training!")

        # Local epoch
        for iter in range(self.args.local_ep):

            total_loss = 0
            total_num = 0

            if 'ldbr' in self.args.baseline:
                iteration = 1
                progress_bar = tqdm(currentBuffer.get_minibatch(self.args.local_bs),
                                    total=length // self.args.local_bs, ncols=100,
                                    disable=not accelerator.is_local_main_process)

                for x, mask, y, t, origin_fea in progress_bar:

                    if iteration % 10 == 0 and self.args.task > 0:
                        # Replay.
                        total_x, total_mask, total_y, total_t, total_fea = x, mask, y, t, origin_fea
                        for j in range(self.args.task):
                            old_x, old_mask, old_y, old_t, old_fea = \
                                buffer.get_random_batch(self.args.local_bs, j)
                            total_x = torch.cat([old_x, total_x], dim=0)
                            total_mask = torch.cat([old_mask, total_mask], dim=0)
                            total_y = torch.cat([old_y, total_y], dim=0)
                            total_t = torch.cat([old_t, total_t], dim=0)
                            total_fea = torch.cat([old_fea, total_fea], dim=0)
                        for j in range(self.args.task + 1):
                            x = total_x[j * self.args.batch_size: (j + 1) * self.args.batch_size, :]
                            mask = total_mask[j * self.args.batch_size: (j + 1) * self.args.batch_size, :]
                            y = total_y[j * self.args.batch_size: (j + 1) * self.args.batch_size]
                            t = total_t[j * self.args.batch_size: (j + 1) * self.args.batch_size]
                            fea = total_fea[j * self.args.batch_size: (j + 1) * self.args.batch_size, :]
                            x, mask, y, t, fea = \
                                x.cuda(), mask.cuda(), y.cuda(), t.cuda(), fea.cuda()
                            loss = ldbr_model.train_step(model, x, mask, y, t, self.args.task, True, fea, predictor)
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer_P.step()
                            optimizer.zero_grad()
                            optimizer_P.zero_grad()

                        iteration += 1
                        global_step += 1
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        progress_bar.set_description('Train Iter (Epoch=%3d,loss=%5.3f)' % (
                            (iter, loss.item())))  # show the loss

                    else:
                        x, mask, y, t, origin_fea = x.cuda(), mask.cuda(), y.cuda(), t.cuda(), origin_fea.cuda()
                        # if self.args.dataset_name == 'tacred':
                        #     import pdb
                        #     pdb.set_trace()
                        loss = \
                            ldbr_model.train_step(model, x, mask, y, t, self.args.task, False, origin_fea,
                                                  predictor)

                        iteration += 1
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer_P.step()

                        global_step += 1
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                        optimizer.zero_grad()
                        optimizer_P.zero_grad()

                        progress_bar.set_description('Train Iter (Epoch=%3d,loss=%5.3f)' % (
                            (iter, loss.item())))  # show the loss

            else:
                progress_bar = tqdm(range(len(train_loader)), disable=not accelerator.is_local_main_process)

                for batch_idx, inputs in enumerate(train_loader):

                    if 'ewc' in self.args.baseline:
                        if 'bart_classification' in self.args.baseline:
                            outputs = model(**inputs, self_fisher=self_fisher)
                        else:
                            outputs = model(inputs, self_fisher=self_fisher)

                    elif 'l2p' in self.args.baseline:
                        outputs = model(**inputs)

                    elif 'experience_replay' in self.args.baseline or 'derpp' in self.args.baseline:
                        if 'bart' in self.args.baseline:
                            outputs = model(**inputs, buffer=buffer)
                        else:
                            outputs = model(inputs, buffer=buffer)

                    elif 'bart_classification' in self.args.baseline:
                        outputs = model(**inputs, restrict_label=True)

                    else:
                        outputs = model(inputs)

                    loss = outputs.loss

                    # 完全不一样了，基于lora的最后一层输出跟预训练model的输出差了好几个量级
                    if 'distill' in self.args.baseline:
                        distill_loss = outputs.distill_loss
                        loss = loss + self.args.lamb_distill * distill_loss

                    accelerator.backward(loss)

                    # if accelerator.is_main_process and iter < 1 and batch_idx < 1:
                    #     for n, p in model.named_parameters():
                    #         if p.grad is not None:
                    #             print('n,p： ', n, p.size())

                    optimizer.step()

                    global_step += 1
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                    optimizer.zero_grad()
                    # current_lr = optimizer.param_groups[0]['lr']
                    # print(current_lr)
                    progress_bar.update(1)
                    progress_bar.set_description(
                        'Train Iter (Epoch=%3d,loss=%5.3f)' % (iter, loss.item()))

                    total_loss += loss.data.cpu().numpy().item() * inputs['input_ids'].size(0)
                    total_num += inputs['input_ids'].size(0)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if dev_loader is None:
                # If we don't use dev set for early stopping, we save the model after the training is finished.
                # self.save_model(accelerator, model)
                #     self.last_model = model
                pass

            # self.tokenizer.save_pretrained(self.args.output_dir)
            if 'ldbr' in self.args.baseline:
                predictor_save_path = os.path.join(current_output_dir, 'predictor.pth')
                torch.save(predictor.state_dict(), predictor_save_path)
                print("select samples to store....")
                ldbr_model.select_samples_to_store(model, buffer, train_loader, current_task, self.args.store_ratio)
                buffer_save_path = os.path.join(current_output_dir, 'buffer.json')
                buffer.save(buffer_save_path)

        if 'ewc' in self.args.baseline:
            fisher_compute_path = os.path.join(current_output_dir, 'fisher')
            os.makedirs(os.path.dirname(fisher_compute_path), exist_ok=True)
            logger.info('Computing fisher matrix for ewc')
            fisher = fisher_model.fisher_compute(train_loader, model, self_fisher, accelerator, self.args)
            torch.save(fisher, fisher_compute_path)
            logger.info(f"Client {idx} Task {current_task}: 保存 Fisher 矩阵到 {fisher_compute_path}")
            # 保存当前任务编号到 last_task.txt
            with open(last_task_path, 'w') as f:
                f.write(str(current_task))
                logger.info(f"Client {idx} Task {current_task}: 任务完成，已保存任务编号到 {last_task_path}")

        elif 'experience_replay' in self.args.baseline:
            # Make sure the random seeds are different when running different tasks. Otherwise, the reservoir sampling
            # is not truly random.
            np.random.seed(self.args.seed * train_loader.dataset['labels'][0].item())
            # Add new data to the buffer and save the new buffer.
            for _, inputs in enumerate(train_loader):
                buffer.add_data(inputs['input_ids'],
                                labels=inputs['labels'],
                                attention_mask=inputs['attention_mask'])
            print(f'The buffer now contains {buffer.num_seen_examples} examples!')
            buffer_save_path = os.path.join(current_output_dir, 'buffer.pth')
            torch.save(buffer, buffer_save_path)

        elif 'derpp' in self.args.baseline:
            # We also need to save the logits.
            model.eval()
            with torch.no_grad():
                for _, inputs in enumerate(train_loader):
                    outputs = model(**inputs)
                    logits = outputs.logits.cpu()
                    buffer.add_data(inputs['input_ids'],
                                    labels=inputs['labels'],
                                    logits=logits,
                                    attention_mask=inputs['attention_mask'])
            print(f'The buffer now contains {buffer.num_seen_examples} examples!')
            buffer_save_path = os.path.join(current_output_dir, 'buffer.pth')
            torch.save(buffer, buffer_save_path)

        model = accelerator.unwrap_model(model)
        model.cpu()
        return model, None
        # return model.state_dict(), None

    def eval(self, model, dataloader, accelerator, task_label_mask=None):
        model.eval()
        label_list = []
        prediction_list = []
        til_prediction_list = []
        total_loss = 0
        total_num = 0
        tid_pred_correct_num = 0
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(len(dataloader)), disable=not accelerator.is_local_main_process)
        with torch.no_grad():
            for batch, inputs in enumerate(dataloader):
                input_ids = inputs['input_ids']

                outputs = model(**inputs)

                real_b = input_ids.size(0)
                loss = outputs.loss
                outp = outputs.logits

                pred = outp.max(1)[1]

                predictions = accelerator.gather(pred)
                references = accelerator.gather(inputs['labels'])

                total_loss += loss.data.cpu().numpy().item() * real_b
                total_num += real_b
                label_list += references.cpu().numpy().tolist()
                prediction_list += predictions.cpu().numpy().tolist()

                # If task_label_mask is known, we can calculate the TIL acc (within-task prediction acc)
                # and the task-id prediction acc.
                if task_label_mask is not None:
                    masked_outp = outputs.logits * task_label_mask.to(outputs.logits.device)
                    til_pred = masked_outp.max(1)[1]
                    til_predictions = accelerator.gather(til_pred)
                    til_prediction_list += til_predictions.cpu().numpy().tolist()
                    for i in predictions:
                        y = i.item()
                        if task_label_mask[y] == 1:  # Predict the task id correctly.
                            tid_pred_correct_num += 1

                progress_bar.update(1)

        micro_f1 = f1_score(label_list, prediction_list, average='micro')
        macro_f1 = f1_score(label_list, prediction_list, average='macro')
        correct_num = sum([float(label_list[i] == prediction_list[i]) for i in range(len(label_list))])
        accuracy = correct_num * 1.0 / len(prediction_list)
        if task_label_mask is not None:
            til_correct_num = sum([float(label_list[i] == til_prediction_list[i]) for i in range(len(label_list))])
            til_accuracy = til_correct_num * 1.0 / len(til_prediction_list)
            tid_pred_accuracy = tid_pred_correct_num * 1.0 / len(til_prediction_list)
        else:
            til_correct_num = -1
            til_accuracy = -1
            tid_pred_correct_num = -1
            tid_pred_accuracy = -1  # Not applicable.

        return micro_f1, macro_f1, accuracy, total_loss / total_num, correct_num, len(prediction_list), \
            prediction_list, label_list, til_accuracy, til_correct_num, tid_pred_accuracy, tid_pred_correct_num

    def get_grad(self, model):

        grad = torch.tensor([]).cuda()  # 在GPU上初始化grad
        param = torch.tensor([]).cuda()  # 在GPU上初始化param

        model_state_dict = model.state_dict()
        global_state_dict = self.global_model.state_dict()

        if self.args.is_peft == 1:
            for layer in global_state_dict:
                if "num_batches_tracked" in layer:
                    continue
                if 'lora' in layer:
                    try:
                        param_now = model_state_dict[layer].detach().cuda()  # 保证在GPU上
                        param_last = global_state_dict[layer].detach().cuda()  # 保证在GPU上
                    except KeyError:
                        raise KeyError(f"无法在 model 和 global_model 的 state_dict 中找到键: '{layer}'")

                    param_g = param_last - param_now  # 计算梯度
                    param = torch.cat((param, param_now.view(-1)))  # 拼接参数
                    grad = torch.cat((grad, param_g.view(-1)))  # 拼接梯度
        else:
            for layer in global_state_dict:
                if "num_batches_tracked" in layer:
                    continue
                try:
                    param_now = model_state_dict[layer].detach().cuda()  # 保证在GPU上
                    param_last = global_state_dict[layer].detach().cuda()  # 保证在GPU上
                except KeyError:
                    raise KeyError(f"无法在 model 和 global_model 的 state_dict 中找到键: '{layer}'")

                param_g = param_last - param_now  # 计算梯度
                param = torch.cat((param, param_now.view(-1)))  # 拼接参数
                grad = torch.cat((grad, param_g.view(-1)))  # 拼接梯度

        # 确保在条件未满足时也有默认值
        grad_cpu = grad.cpu() if grad.numel() > 0 else grad  # 如果 grad 非空，移到CPU
        param_cpu = param.cpu() if param.numel() > 0 else param  # 如果 param 非空，移到CPU

        # 显式删除GPU上的张量，释放内存
        del grad
        del param

        # 调用空缓存
        torch.cuda.empty_cache()

        return grad_cpu, param_cpu

    def aggregate(self, grad_dist: dict, cohorts: list, partition_map: dict):
        print("Begin aggregate...")
        model_gra = torch.zeros_like(grad_dist[cohorts[0]])
        data_sum = 0
        for client in cohorts:
            data_sum += len(partition_map[client])
        for client in cohorts:
            w = len(partition_map[client]) / data_sum
            model_gra += (w * grad_dist[client])
        global_model = self.combine(grad=model_gra)
        return model_gra, global_model

    def combine(self, grad, gobal_model=None):
        if gobal_model is None:
            gobal_model = self.global_model
        grad = grad.cuda()
        current_index = 0
        model = copy.deepcopy(gobal_model)
        current_state_dict = model.state_dict()

        if self.args.is_peft == 1:
            for name, param in current_state_dict.items():
                if 'lora' in name:
                    param = param.cuda()
                    numel = param.data.numel()
                    size = param.data.size()
                    current_state_dict[name] = \
                        torch.subtract(param.data.detach(), grad[current_index:current_index + numel].view(size))
                    current_index += numel
        else:
            for name, param in current_state_dict.items():
                param = param.cuda()
                numel = param.data.numel()
                size = param.data.size()
                current_state_dict[name] = \
                    torch.subtract(param.data.detach(), grad[current_index:current_index + numel].view(size))
                current_index += numel

        model.load_state_dict(current_state_dict)

        return model
