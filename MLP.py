import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt # for making figures
%matplotlib inline

class MLP:
    def __init__(self, words, context_size, layer_size, embedment_dims, learning_rate, revolutions):
        self.build_vocab(words)
        self.initialize_fields(words, context_size, layer_size, embedment_dims, learning_rate, revolutions)
        self.initialize_datasets()
        self.init_params()
        self.initialize_graphing_fields()

    def build_vocab(self, words):
        chars = sorted(list(set(''.join(words))))
        self.stoi = {s: i + 1 for i, s in enumerate(chars)}
        self.stoi['.'] = 0
        self.itos = {i: s for s, i in self.stoi.items()}

    def initialize_fields(self, words, context_size, layer_size, embedment_dims, learning_rate, revolutions):
        self.words = words
        self.context_size = torch.nn.Parameter(torch.tensor(float(context_size), requires_grad=True))
        self.layer_size = torch.nn.Parameter(torch.tensor(float(layer_size), requires_grad=True))
        self.embedment_dims = torch.nn.Parameter(torch.tensor(float(embedment_dims), requires_grad=True))
        self.learning_rate = torch.nn.Parameter(torch.tensor(float(learning_rate), requires_grad=True))
        self.revolutions = torch.nn.Parameter(torch.tensor(float(revolutions), requires_grad=True))
        self.generator = torch.Generator().manual_seed(2147483647)  # for reproducibility

    def initialize_datasets(self):
        n1 = int(0.8 * len(self.words))
        n2 = int(0.9 * len(self.words))
        self.Xtr, self.Ytr = self.build_dataset(self.words[:n1])
        self.Xdev, self.Ydev = self.build_dataset(self.words[n1:n2])
        self.Xte, self.Yte = self.build_dataset(self.words[n2:])

    def build_dataset(self, words):
        X, Y = [], []
        for w in words:
            context = [0] * int(self.context_size.item())
            for ch in w + '.':
                ix = self.stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X, Y

    def init_params(self):
        self.C = torch.randn([27, int(self.embedment_dims.item())], generator=self.generator)
        self.W1 = torch.rand((int(self.embedment_dims.item())) * int(self.context_size.item()), int(self.layer_size.item()),
                             generator=self.generator)
        self.b1 = torch.randn(int(self.layer_size.item()), generator=self.generator)
        self.W2 = torch.rand((int(self.layer_size.item()), 27), generator=self.generator)
        self.b2 = torch.randn(27, generator=self.generator)
        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]
        for p in self.parameters:
            p.requires_grad = True
        self.optimizer = optim.SGD([self.context_size, self.layer_size, self.embedment_dims,
                                    self.learning_rate, self.revolutions], lr=0.01)

    def initialize_graphing_fields(self):
        self.lri = []
        self.lossi = []
        self.stepi = []

    def train_stage1(self):
        self.train_dataset(self.revolutions, self.Xtr, self.Ytr)

    def train_stage2(self):
        self.train_dataset(self.revolutions / 2, self.Xdev, self.Ydev)

    def train_stage3(self):
        self.train_dataset(self.revolutions / 4, self.Xte, self.Yte)

    def train_dataset(self, revolutions, Xset, Yset):
        icount = 0
        for i in range(int(revolutions.item())):
            icount += 1
            restarts = 0
            ix = torch.randint(0, Xset.shape[0], (32,))

            emb = self.C[Xset[ix]]
            h = torch.tanh(emb.view(-1, int(self.embedment_dims.item()) * int(self.context_size.item())) @ self.W1 + self.b1)
            logits = h @ self.W2 + self.b2
            loss = F.cross_entropy(logits, Yset[ix])

            for p in self.parameters:
                p.grad = None
            loss.backward()

            if icount == 100000:
                self.learning_rate.data *= (10 ** (-1 * restarts))
                icount = 0
                restarts += 1
            for p in self.parameters:
                p.data += -self.learning_rate * p.grad

            self.stepi.append(i)
            self.lossi.append(loss.log10().item())
        # print(loss.item())

    def train_hps(self):
        new_model1 = MLP(self.words, int(self.context_size.item()), int(self.layer_size.item()), int(self.embedment_dims.item()), int(self.learning_rate.item()), int(self.revolutions.item()))
        new_model1.train_stage1()
        new_model1.train_stage2()
        new_model1.train_stage3()

        old_loss = new_model1.get_test_loss().item()
        for i in range(10):
            # Forward and backward pass to compute gradients
            self.optimizer.zero_grad()
            new_model2 = MLP(self.words, int(self.context_size.item()), int(self.layer_size.item()) - 1, int(self.embedment_dims.item()) - 1, int(self.learning_rate.item()), int(self.revolutions.item()))
            print(new_model2.layer_size.item())
            print(new_model2.embedment_dims.item())
            print(new_model2.learning_rate.item())

            new_model2.train_stage1()
            new_model2.train_stage2()
            new_model2.train_stage3()
            new_loss = new_model2.get_test_loss().item()

            print("loss1: " + str(old_loss) , "loss2: " + str(new_loss))
            loss_diff = (old_loss - new_loss) / 10
            print("loss_diff:" + str(loss_diff))
    
            # Update hyperparameters using gradient descent
            self.layer_size.data -= loss_diff * self.layer_size.data
            self.embedment_dims.data -= loss_diff * self.embedment_dims.data
            self.learning_rate.data -= loss_diff * self.learning_rate.data

            # Reinitialize parameters with updated hyperparameters
            self.init_params()

            # Zero gradients for the next iteration
            self.optimizer.zero_grad()

            old_loss = new_loss

            if self.layer_size.data == 1.0:
                break

            if self.embedment_dims.data == 1.0:
                break

    def get_total_params(self):
        print(sum(p.nelement() for p in self.parameters))

    def plt_loss(self):
        plt.plot(self.stepi, self.lossi)

    def plt_embeddings(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.C[:, 0].data, self.C[:, 1].data, s=200)
        for i in range(self.C.shape[0]):
            plt.text(self.C[i, 0].item(), self.C[i, 1].item(), self.itos[i], ha="center", va="center", color="white")
        plt.grid('minor')

    def print_training_loss(self):
        emb = self.C[self.Xtr]
        h = torch.tanh(emb.view(-1, int(self.embedment_dims.item()) * int(self.context_size)) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        print(F.cross_entropy(logits, self.Ytr))

    def get_training_loss(self):
        emb = self.C[self.Xtr]
        h = torch.tanh(emb.view(-1, int(self.embedment_dims.item()) * int(self.context_size)) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return F.cross_entropy(logits, self.Ytr)

    def print_dev_loss(self):
        emb = self.C[self.Xdev]
        h = torch.tanh(emb.view(-1, int(self.embedment_dims.item()) * int(self.context_size.item())) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        print(F.cross_entropy(logits, self.Ydev))

    def get_dev_loss(self):
        emb = self.C[self.Xdev]
        h = torch.tanh(emb.view(-1, int(self.embedment_dims.item()) * int(self.context_size.item())) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return F.cross_entropy(logits, self.Ydev)

    def print_test_loss(self):
        emb = self.C[self.Xte]
        h = torch.tanh(emb.view(-1, int(self.embedment_dims.item()) * int(self.context_size.item())) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        print(F.cross_entropy(logits, self.Yte))

    def get_test_loss(self):
        emb = self.C[self.Xte]
        h = torch.tanh(emb.view(-1, int(self.embedment_dims.item()) * int(self.context_size.item())) @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return F.cross_entropy(logits, self.Yte)

    def sample_model(self):
        g = torch.Generator().manual_seed(2147483647 + 10)

        for _ in range(20):
            out = []
            context = [0] * int(self.context_size.item())
            while True:
                emb = self.C[torch.tensor([context])]
                h = torch.tanh(emb.view(1, -1) @ self.W1 + self.b1)
                logits = h @ self.W2 + self.b2
                probs = F.softmax(logits, dim=1)
                ix = torch.multinomial(probs, num_samples=1, generator=g).item()
                context = context[1:] + [ix]
                out.append(ix)
                if ix == 0:
                    break

            print(''.join(self.itos[i] for i in out))

words = open('names.txt', 'r').read().splitlines()
model = MLP(words, 5, 163, 5, 0.0954, 500000)
model.get_test_loss()
