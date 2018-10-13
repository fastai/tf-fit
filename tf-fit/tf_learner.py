import tensorflow as tf
from fastai import *

tf.enable_eager_execution()
tfe = tf.contrib.eager
tf.keras.backend.set_image_data_format('channels_first')

__all__ = []

tf_flatten_model=lambda m: sum(map(tf_flatten_model,m.layers),[]) if hasattr(m, "layers") else [m]

tf_bn_types = (tf.keras.layers.BatchNormalization,)

tf.Tensor.detach = lambda x: x.numpy()


def tf_loss_batch(model, xb, yb, loss_fn=None, opt=None, cb_handler=None, metrics=None):
    "Calculate loss and metrics for a batch, call out to callbacks as necessary."
    if cb_handler is None: cb_handler = CallbackHandler([])
    if not is_listy(xb): xb = [xb]
    if not is_listy(yb): yb = [yb]
    
    xb = [tf.constant(v.cpu().numpy()) for v in xb]
    yb = [tf.constant(v.cpu().numpy()) for v in yb]
    
    
    
    def forward():
        out = model(*xb)
        out = cb_handler.on_loss_begin(out)
        return out
    def forward_calc_loss_mets():
        out = forward()
        loss = loss_fn(*yb, out) #reversed params compared to pytorch
        mets = [f(*yb, out).numpy() for f in metrics] if metrics is not None else [] #note metrics params order
        loss = cb_handler.on_backward_begin(loss)
        return loss, mets
    
    
    if not loss_fn:
        out = forward()
        return out,yb[0]
        
    loss, mets = None, None
    if opt is not None:
        with tf.GradientTape() as tape:
            loss, mets = forward_calc_loss_mets()
            
        
        grads = tape.gradient(loss, model.weights)
        cb_handler.on_backward_end()
        opt.apply_gradients(zip(grads, model.weights))
        cb_handler.on_step_end()
        
    else:
        loss, mets = forward_calc_loss_mets()
        
    
    return ((loss.numpy(),) + tuple(mets))
    
def tf_get_preds(model, dl, pbar=None, cb_handler=None):
    "Predict the output of the elements in the dataloader."
    return [np.concatenate(o) for o in zip(*tf_validate(model, dl, pbar=pbar, cb_handler=cb_handler, average=False))]
    
    
def tf_validate(model, dl, loss_fn=None, metrics=None, cb_handler=None, pbar=None, average=True):
    "Calculate loss and metrics for the validation set."
    
    val_metrics,nums = [],[]
    for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
        if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
        val_metrics.append(tf_loss_batch(model, xb, yb, loss_fn, cb_handler=cb_handler, metrics=metrics))
        if not is_listy(yb): yb = [yb]
        nums.append(yb[0].shape[0])
        if cb_handler and cb_handler.on_batch_end(val_metrics[0]): break
    nums = np.array(nums, dtype=np.float32)
    if average: return [(np.stack(val) * nums).sum() / nums.sum() for val in zip(*val_metrics)]
    else: return val_metrics


def tf_train_epoch(model, dl, opt, loss_func):
    "Simple training of `model` for 1 epoch of `dl` using optim `opt` and loss function `loss_func`."
    for xb,yb in dl:
        if not is_listy(xb): xb = [xb]
        if not is_listy(yb): yb = [yb]

        xb = [tf.constant(v.cpu().numpy()) for v in xb]
        yb = [tf.constant(v.cpu().numpy()) for v in yb]

        with tf.GradientTape() as tape:
            out = model(*xb)
            loss = loss_fn(*yb, out) #reversed params compared to pytorch


        grads = tape.gradient(loss, model.weights)
        opt.apply_gradients(zip(grads, model.weights))

def tf_fit(epochs, model, loss_fn, opt, data, callbacks, metrics):
    cb_handler = CallbackHandler(callbacks)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception=False
    try:
        for epoch in pbar:
            cb_handler.on_epoch_begin()

            for xb,yb in progress_bar(data.train_dl, parent=pbar):
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = tf_loss_batch(model, xb, yb, loss_fn, opt, cb_handler)
                if cb_handler.on_batch_end(loss): break

            if hasattr(data,'valid_dl') and data.valid_dl is not None:
                val_metrics = tf_validate(model, data.valid_dl, loss_fn=loss_fn,
                                       cb_handler=cb_handler, metrics=metrics,pbar=pbar)

            else: val_metrics=None
            if cb_handler.on_epoch_end(val_metrics): break
    except Exception as e:
        exception = e
        raise e
    finally: cb_handler.on_train_end(exception)

    
    
    
    
@dataclass
class TfLearner():
    "Train `model` using `data` to minimize `loss_fn` with optimizer `opt_fn`."
    data:DataBunch
    model:tf.keras.Model
    opt_fn:Callable
    loss_fn:Callable
    metrics:Collection[Callable]=None
    true_wd:bool=True
    bn_wd:bool=True
    wd:float=default_wd
    train_bn:bool=True
    path:str=None
    model_dir:str='models'
    callback_fns:Collection[Callable]=None
    callbacks:Collection[Callback]=field(default_factory=list)
    layer_groups:Collection[tf.keras.layers.Layer]=None
    def __post_init__(self)->None:
        "Setup path,metrics, callbacks and ensure model directory exists."
        self.path = Path(ifnone(self.path, self.data.path))
        (self.path/self.model_dir).mkdir(parents=True, exist_ok=True)
        self.metrics=listify(self.metrics)
        if not self.layer_groups: self.layer_groups = tf_flatten_model(self.model)
        self.callbacks = listify(self.callbacks)
        self.callback_fns = [Recorder] + [TfRegularizer] + listify(self.callback_fns)
        
        #build the model by running 1 batch
        xb, yb = next(iter(self.data.train_dl))
        tf_loss_batch(self.model, xb, yb)

    def init(self, init): raise NotImplementedError

    def lr_range(self, lr:Union[float,slice])->np.ndarray:
        "Build differential learning rates."
        if not isinstance(lr,slice): return lr
        if lr.start: res = even_mults(lr.start, lr.stop, len(self.layer_groups))
        else: res = [lr.stop/3]*(len(self.layer_groups)-1) + [lr.stop]
        return np.array(res)

    def fit(self, epochs:int, lr:Union[Floats,slice]=default_lr,
            wd:Floats=None, callbacks:Collection[Callback]=None)->None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None: wd = self.wd
        self.create_opt(lr, wd)
        callbacks = [cb(self) for cb in self.callback_fns] + listify(callbacks)      
        tf_fit(epochs, self.model, self.loss_fn, opt=self.opt, data=self.data, metrics=self.metrics,
            callbacks=self.callbacks+callbacks)

    def create_opt(self, lr:Floats, wd:Floats=0.)->None:
        "Create optimizer with `lr` learning rate and `wd` weight decay."
        self.opt = TfOptimWrapper.create(self.opt_fn, lr, wd, self.layer_groups)


    def freeze_to(self, n:int)->None:
        "Freeze layers up to layer `n`."
        for l in self.layer_groups[:n]: 
            
            if not self.train_bn or not isinstance(l, bn_types): l.trainable = False
                
        for l in self.layer_groups[n:]: l.trainable = True

    def freeze(self)->None:
        "Freeze up to last layer."
        assert(len(self.layer_groups)>1)
        self.freeze_to(-1)

    def unfreeze(self):
        "Unfreeze entire model."
        self.freeze_to(0)

    def __del__(self): del(self.model, self.data)

    def save(self, name:PathOrStr):
        "Save model with `name` to `self.model_dir`."
        root = tf.train.Checkpoint(model=self.model)
        root.save(file_prefix=self.path/self.model_dir/f'{name}')


    def load(self, name:PathOrStr):
        "Load model `name` from `self.model_dir`."
        root = tf.train.Checkpoint(model=self.model)
        root.restore(str(self.path/self.model_dir/f'{name}-1'))
    
    def get_preds(self, is_test=False):
        return tf_get_preds(self.model, self.data.holdout(is_test), cb_handler=CallbackHandler(self.callbacks))  



TfLearner.fit_one_cycle = fit_one_cycle
TfLearner.lr_find = lr_find






class TfOptimWrapper():
    def __init__(self, opt_fn, layer_groups):
        self.layer_groups = layer_groups
        self._lr = [tfe.Variable(0.0) for o in layer_groups]
        self._mom = tfe.Variable(0.0)
        self._wd = 0.0
        
        
        opt_params = inspect.signature(opt_fn).parameters
        params = {}
        if opt_params.get("momentum"):
            self.mom = opt_params.get("momentum").default
            params["momentum"] = self._mom
        if opt_params.get("beta1"):
            self.mom = opt_params.get("beta1").default
            params["beta1"] = self._mom
        
        
        self.opt = [opt_fn(learning_rate=o, **params) for o in self._lr]
        
        
    @classmethod
    def create(cls, opt_fn, lr, wd, layer_groups, **kwargs):
        opt = cls(opt_fn, layer_groups,  **kwargs)
        
        
        opt.lr = lr
        opt.wd = wd
        return opt
    
        
    def apply_gradients(self, grads_and_vars):
        for gv, opt, l in zip(grads_and_vars, self.opt, self.layer_groups):
            if l.trainable: opt.apply_gradients([gv])
        
    
    @property
    def lr(self)->float:
        "Get learning rate."
        return self._lr[-1].numpy()

    @lr.setter
    def lr(self, val:float)->None:
        "Set learning rate."
        val = listify(val, self._lr)
        for o, v in zip(self._lr, val): o.assign(v) 

    @property
    def mom(self)->float:
        "Get momentum."
        return self._mom.numpy()

    @mom.setter
    def mom(self, val:float)->None:
        "Set momentum."
        if not isinstance(val, float): val = val[-1]
        self._mom.assign(val)
        
        
    @property
    def wd(self)->float:
        "Get wd."
        return self._wd

    @wd.setter
    def wd(self, val:float)->None:
        "Set wd."
        self._wd = val






class TfRegularizer(LearnerCallback):
    def __init__(self, learn:Learner):
        super().__init__(learn)
    #true_wd=False
    def on_backward_begin(self, last_loss, **kwargs):
        if not self.learn.true_wd:
            regularizer = sum([tf.nn.l2_loss(w) for w in self.learn.model.weights])
            return last_loss + self.learn.wd * regularizer
    #true_wd=True
    def on_backward_end(self, **kwargs):
        if self.learn.true_wd:
            opt = self.learn.opt
            for lr, l in zip(opt._lr, opt.layer_groups):
                if l.trainable:
                    if self.learn.bn_wd or not isinstance(l, tf_bn_types):
                        for w in l.weights: w=w*lr*opt.wd