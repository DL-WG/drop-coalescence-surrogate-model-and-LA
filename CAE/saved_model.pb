??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12unknown8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
p
kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namekernel
i
kernel/Read/ReadVariableOpReadVariableOpkernel*&
_output_shapes
:*
dtype0
`
biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias
Y
bias/Read/ReadVariableOpReadVariableOpbias*
_output_shapes
:*
dtype0
t
kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
kernel_1
m
kernel_1/Read/ReadVariableOpReadVariableOpkernel_1*&
_output_shapes
:*
dtype0
d
bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_1
]
bias_1/Read/ReadVariableOpReadVariableOpbias_1*
_output_shapes
:*
dtype0
t
kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
kernel_2
m
kernel_2/Read/ReadVariableOpReadVariableOpkernel_2*&
_output_shapes
: *
dtype0
d
bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebias_2
]
bias_2/Read/ReadVariableOpReadVariableOpbias_2*
_output_shapes
: *
dtype0
m
kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name
kernel_3
f
kernel_3/Read/ReadVariableOpReadVariableOpkernel_3*
_output_shapes
:	?*
dtype0
d
bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_3
]
bias_3/Read/ReadVariableOpReadVariableOpbias_3*
_output_shapes
:*
dtype0
m
kernel_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_name
kernel_4
f
kernel_4/Read/ReadVariableOpReadVariableOpkernel_4*
_output_shapes
:	?*
dtype0
e
bias_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namebias_4
^
bias_4/Read/ReadVariableOpReadVariableOpbias_4*
_output_shapes	
:?*
dtype0
t
kernel_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:  *
shared_name
kernel_5
m
kernel_5/Read/ReadVariableOpReadVariableOpkernel_5*&
_output_shapes
:  *
dtype0
d
bias_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebias_5
]
bias_5/Read/ReadVariableOpReadVariableOpbias_5*
_output_shapes
: *
dtype0
t
kernel_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
kernel_6
m
kernel_6/Read/ReadVariableOpReadVariableOpkernel_6*&
_output_shapes
: *
dtype0
d
bias_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_6
]
bias_6/Read/ReadVariableOpReadVariableOpbias_6*
_output_shapes
:*
dtype0
t
kernel_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
kernel_7
m
kernel_7/Read/ReadVariableOpReadVariableOpkernel_7*&
_output_shapes
:*
dtype0
d
bias_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_7
]
bias_7/Read/ReadVariableOpReadVariableOpbias_7*
_output_shapes
:*
dtype0
t
kernel_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
kernel_8
m
kernel_8/Read/ReadVariableOpReadVariableOpkernel_8*&
_output_shapes
:*
dtype0
d
bias_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namebias_8
]
bias_8/Read/ReadVariableOpReadVariableOpbias_8*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
~
Adam/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/kernel/m
w
!Adam/kernel/m/Read/ReadVariableOpReadVariableOpAdam/kernel/m*&
_output_shapes
:*
dtype0
n
Adam/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m
g
Adam/bias/m/Read/ReadVariableOpReadVariableOpAdam/bias/m*
_output_shapes
:*
dtype0
?
Adam/kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/kernel/m_1
{
#Adam/kernel/m_1/Read/ReadVariableOpReadVariableOpAdam/kernel/m_1*&
_output_shapes
:*
dtype0
r
Adam/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m_1
k
!Adam/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/bias/m_1*
_output_shapes
:*
dtype0
?
Adam/kernel/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/kernel/m_2
{
#Adam/kernel/m_2/Read/ReadVariableOpReadVariableOpAdam/kernel/m_2*&
_output_shapes
: *
dtype0
r
Adam/bias/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/bias/m_2
k
!Adam/bias/m_2/Read/ReadVariableOpReadVariableOpAdam/bias/m_2*
_output_shapes
: *
dtype0
{
Adam/kernel/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameAdam/kernel/m_3
t
#Adam/kernel/m_3/Read/ReadVariableOpReadVariableOpAdam/kernel/m_3*
_output_shapes
:	?*
dtype0
r
Adam/bias/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m_3
k
!Adam/bias/m_3/Read/ReadVariableOpReadVariableOpAdam/bias/m_3*
_output_shapes
:*
dtype0
{
Adam/kernel/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameAdam/kernel/m_4
t
#Adam/kernel/m_4/Read/ReadVariableOpReadVariableOpAdam/kernel/m_4*
_output_shapes
:	?*
dtype0
s
Adam/bias/m_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/bias/m_4
l
!Adam/bias/m_4/Read/ReadVariableOpReadVariableOpAdam/bias/m_4*
_output_shapes	
:?*
dtype0
?
Adam/kernel/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameAdam/kernel/m_5
{
#Adam/kernel/m_5/Read/ReadVariableOpReadVariableOpAdam/kernel/m_5*&
_output_shapes
:  *
dtype0
r
Adam/bias/m_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/bias/m_5
k
!Adam/bias/m_5/Read/ReadVariableOpReadVariableOpAdam/bias/m_5*
_output_shapes
: *
dtype0
?
Adam/kernel/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/kernel/m_6
{
#Adam/kernel/m_6/Read/ReadVariableOpReadVariableOpAdam/kernel/m_6*&
_output_shapes
: *
dtype0
r
Adam/bias/m_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m_6
k
!Adam/bias/m_6/Read/ReadVariableOpReadVariableOpAdam/bias/m_6*
_output_shapes
:*
dtype0
?
Adam/kernel/m_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/kernel/m_7
{
#Adam/kernel/m_7/Read/ReadVariableOpReadVariableOpAdam/kernel/m_7*&
_output_shapes
:*
dtype0
r
Adam/bias/m_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m_7
k
!Adam/bias/m_7/Read/ReadVariableOpReadVariableOpAdam/bias/m_7*
_output_shapes
:*
dtype0
?
Adam/kernel/m_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/kernel/m_8
{
#Adam/kernel/m_8/Read/ReadVariableOpReadVariableOpAdam/kernel/m_8*&
_output_shapes
:*
dtype0
r
Adam/bias/m_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/m_8
k
!Adam/bias/m_8/Read/ReadVariableOpReadVariableOpAdam/bias/m_8*
_output_shapes
:*
dtype0
~
Adam/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/kernel/v
w
!Adam/kernel/v/Read/ReadVariableOpReadVariableOpAdam/kernel/v*&
_output_shapes
:*
dtype0
n
Adam/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v
g
Adam/bias/v/Read/ReadVariableOpReadVariableOpAdam/bias/v*
_output_shapes
:*
dtype0
?
Adam/kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/kernel/v_1
{
#Adam/kernel/v_1/Read/ReadVariableOpReadVariableOpAdam/kernel/v_1*&
_output_shapes
:*
dtype0
r
Adam/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v_1
k
!Adam/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/bias/v_1*
_output_shapes
:*
dtype0
?
Adam/kernel/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/kernel/v_2
{
#Adam/kernel/v_2/Read/ReadVariableOpReadVariableOpAdam/kernel/v_2*&
_output_shapes
: *
dtype0
r
Adam/bias/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/bias/v_2
k
!Adam/bias/v_2/Read/ReadVariableOpReadVariableOpAdam/bias/v_2*
_output_shapes
: *
dtype0
{
Adam/kernel/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameAdam/kernel/v_3
t
#Adam/kernel/v_3/Read/ReadVariableOpReadVariableOpAdam/kernel/v_3*
_output_shapes
:	?*
dtype0
r
Adam/bias/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v_3
k
!Adam/bias/v_3/Read/ReadVariableOpReadVariableOpAdam/bias/v_3*
_output_shapes
:*
dtype0
{
Adam/kernel/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_nameAdam/kernel/v_4
t
#Adam/kernel/v_4/Read/ReadVariableOpReadVariableOpAdam/kernel/v_4*
_output_shapes
:	?*
dtype0
s
Adam/bias/v_4VarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameAdam/bias/v_4
l
!Adam/bias/v_4/Read/ReadVariableOpReadVariableOpAdam/bias/v_4*
_output_shapes	
:?*
dtype0
?
Adam/kernel/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameAdam/kernel/v_5
{
#Adam/kernel/v_5/Read/ReadVariableOpReadVariableOpAdam/kernel/v_5*&
_output_shapes
:  *
dtype0
r
Adam/bias/v_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/bias/v_5
k
!Adam/bias/v_5/Read/ReadVariableOpReadVariableOpAdam/bias/v_5*
_output_shapes
: *
dtype0
?
Adam/kernel/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameAdam/kernel/v_6
{
#Adam/kernel/v_6/Read/ReadVariableOpReadVariableOpAdam/kernel/v_6*&
_output_shapes
: *
dtype0
r
Adam/bias/v_6VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v_6
k
!Adam/bias/v_6/Read/ReadVariableOpReadVariableOpAdam/bias/v_6*
_output_shapes
:*
dtype0
?
Adam/kernel/v_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/kernel/v_7
{
#Adam/kernel/v_7/Read/ReadVariableOpReadVariableOpAdam/kernel/v_7*&
_output_shapes
:*
dtype0
r
Adam/bias/v_7VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v_7
k
!Adam/bias/v_7/Read/ReadVariableOpReadVariableOpAdam/bias/v_7*
_output_shapes
:*
dtype0
?
Adam/kernel/v_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameAdam/kernel/v_8
{
#Adam/kernel/v_8/Read/ReadVariableOpReadVariableOpAdam/kernel/v_8*&
_output_shapes
:*
dtype0
r
Adam/bias/v_8VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameAdam/bias/v_8
k
!Adam/bias/v_8/Read/ReadVariableOpReadVariableOpAdam/bias/v_8*
_output_shapes
:*
dtype0

NoOpNoOp
?u
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?t
value?tB?t B?t
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer

signatures
#_self_saveable_object_factories
trainable_variables
	variables
	regularization_losses

	keras_api
%
#_self_saveable_object_factories
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
 layer_with_weights-3
 layer-7
!layer-8
"layer_with_weights-4
"layer-9
##_self_saveable_object_factories
$trainable_variables
%	variables
&regularization_losses
'	keras_api
?
(iter

)beta_1

*beta_2
	+decay
,learning_rate-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v?
 
 
?
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
?
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
 
?
?metrics
@layer_regularization_losses
Alayer_metrics
trainable_variables
	variables

Blayers
Cnon_trainable_variables
	regularization_losses
 
?

-kernel
.bias
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
w
#I_self_saveable_object_factories
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
?

/kernel
0bias
#N_self_saveable_object_factories
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
w
#S_self_saveable_object_factories
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
?

1kernel
2bias
#X_self_saveable_object_factories
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
w
#]_self_saveable_object_factories
^regularization_losses
_trainable_variables
`	variables
a	keras_api
w
#b_self_saveable_object_factories
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?

3kernel
4bias
#g_self_saveable_object_factories
hregularization_losses
itrainable_variables
j	variables
k	keras_api
 
8
-0
.1
/2
03
14
25
36
47
8
-0
.1
/2
03
14
25
36
47
 
?
lmetrics
mlayer_regularization_losses
nlayer_metrics
trainable_variables
	variables

olayers
pnon_trainable_variables
regularization_losses
%
#q_self_saveable_object_factories
?

5kernel
6bias
#r_self_saveable_object_factories
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
w
#w_self_saveable_object_factories
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
?

7kernel
8bias
#|_self_saveable_object_factories
}regularization_losses
~trainable_variables
	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?

9kernel
:bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?

;kernel
<bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
|
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?

=kernel
>bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
 
F
50
61
72
83
94
:5
;6
<7
=8
>9
F
50
61
72
83
94
:5
;6
<7
=8
>9
 
?
?metrics
 ?layer_regularization_losses
?layer_metrics
$trainable_variables
%	variables
?layers
?non_trainable_variables
&regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEkernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEbias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEkernel_10trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbias_10trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEkernel_20trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbias_20trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEkernel_30trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbias_30trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEkernel_40trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEbias_40trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEkernel_51trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEbias_51trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEkernel_61trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEbias_61trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEkernel_71trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEbias_71trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEkernel_81trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEbias_81trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE

?0
 
 

0
1
2
 
 
 

-0
.1

-0
.1
?
Eregularization_losses
?layer_metrics
?layers
Ftrainable_variables
G	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
Jregularization_losses
?layer_metrics
?layers
Ktrainable_variables
L	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 

/0
01

/0
01
?
Oregularization_losses
?layer_metrics
?layers
Ptrainable_variables
Q	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
Tregularization_losses
?layer_metrics
?layers
Utrainable_variables
V	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 

10
21

10
21
?
Yregularization_losses
?layer_metrics
?layers
Ztrainable_variables
[	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
^regularization_losses
?layer_metrics
?layers
_trainable_variables
`	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
cregularization_losses
?layer_metrics
?layers
dtrainable_variables
e	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 

30
41

30
41
?
hregularization_losses
?layer_metrics
?layers
itrainable_variables
j	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
?
0
1
2
3
4
5
6
7
8
 
 
 
 

50
61

50
61
?
sregularization_losses
?layer_metrics
?layers
ttrainable_variables
u	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
xregularization_losses
?layer_metrics
?layers
ytrainable_variables
z	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 

70
81

70
81
?
}regularization_losses
?layer_metrics
?layers
~trainable_variables
	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 

90
:1

90
:1
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 

;0
<1

;0
<1
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
 
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 

=0
>1

=0
>1
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
 
 
 
F
0
1
2
3
4
5
6
 7
!8
"9
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
om
VARIABLE_VALUEAdam/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/m_1Ltrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/m_1Ltrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/m_2Ltrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/m_2Ltrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/m_3Ltrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/m_3Ltrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/m_4Ltrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/m_4Ltrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/m_5Mtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/m_5Mtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/m_6Mtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/m_6Mtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/m_7Mtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/m_7Mtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/m_8Mtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/m_8Mtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/v_1Ltrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/v_1Ltrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/v_2Ltrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/v_2Ltrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/v_3Ltrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/v_3Ltrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUEAdam/kernel/v_4Ltrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUEAdam/bias/v_4Ltrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/v_5Mtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/v_5Mtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/v_6Mtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/v_6Mtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/v_7Mtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/v_7Mtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEAdam/kernel/v_8Mtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEAdam/bias/v_8Mtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1kernelbiaskernel_1bias_1kernel_2bias_2kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_7bias_7kernel_8bias_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*4
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *.
f)R'
%__inference_signature_wrapper_1593491
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpkernel/Read/ReadVariableOpbias/Read/ReadVariableOpkernel_1/Read/ReadVariableOpbias_1/Read/ReadVariableOpkernel_2/Read/ReadVariableOpbias_2/Read/ReadVariableOpkernel_3/Read/ReadVariableOpbias_3/Read/ReadVariableOpkernel_4/Read/ReadVariableOpbias_4/Read/ReadVariableOpkernel_5/Read/ReadVariableOpbias_5/Read/ReadVariableOpkernel_6/Read/ReadVariableOpbias_6/Read/ReadVariableOpkernel_7/Read/ReadVariableOpbias_7/Read/ReadVariableOpkernel_8/Read/ReadVariableOpbias_8/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp!Adam/kernel/m/Read/ReadVariableOpAdam/bias/m/Read/ReadVariableOp#Adam/kernel/m_1/Read/ReadVariableOp!Adam/bias/m_1/Read/ReadVariableOp#Adam/kernel/m_2/Read/ReadVariableOp!Adam/bias/m_2/Read/ReadVariableOp#Adam/kernel/m_3/Read/ReadVariableOp!Adam/bias/m_3/Read/ReadVariableOp#Adam/kernel/m_4/Read/ReadVariableOp!Adam/bias/m_4/Read/ReadVariableOp#Adam/kernel/m_5/Read/ReadVariableOp!Adam/bias/m_5/Read/ReadVariableOp#Adam/kernel/m_6/Read/ReadVariableOp!Adam/bias/m_6/Read/ReadVariableOp#Adam/kernel/m_7/Read/ReadVariableOp!Adam/bias/m_7/Read/ReadVariableOp#Adam/kernel/m_8/Read/ReadVariableOp!Adam/bias/m_8/Read/ReadVariableOp!Adam/kernel/v/Read/ReadVariableOpAdam/bias/v/Read/ReadVariableOp#Adam/kernel/v_1/Read/ReadVariableOp!Adam/bias/v_1/Read/ReadVariableOp#Adam/kernel/v_2/Read/ReadVariableOp!Adam/bias/v_2/Read/ReadVariableOp#Adam/kernel/v_3/Read/ReadVariableOp!Adam/bias/v_3/Read/ReadVariableOp#Adam/kernel/v_4/Read/ReadVariableOp!Adam/bias/v_4/Read/ReadVariableOp#Adam/kernel/v_5/Read/ReadVariableOp!Adam/bias/v_5/Read/ReadVariableOp#Adam/kernel/v_6/Read/ReadVariableOp!Adam/bias/v_6/Read/ReadVariableOp#Adam/kernel/v_7/Read/ReadVariableOp!Adam/bias/v_7/Read/ReadVariableOp#Adam/kernel/v_8/Read/ReadVariableOp!Adam/bias/v_8/Read/ReadVariableOpConst*J
TinC
A2?	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *)
f$R"
 __inference__traced_save_1594503
?	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratekernelbiaskernel_1bias_1kernel_2bias_2kernel_3bias_3kernel_4bias_4kernel_5bias_5kernel_6bias_6kernel_7bias_7kernel_8bias_8totalcountAdam/kernel/mAdam/bias/mAdam/kernel/m_1Adam/bias/m_1Adam/kernel/m_2Adam/bias/m_2Adam/kernel/m_3Adam/bias/m_3Adam/kernel/m_4Adam/bias/m_4Adam/kernel/m_5Adam/bias/m_5Adam/kernel/m_6Adam/bias/m_6Adam/kernel/m_7Adam/bias/m_7Adam/kernel/m_8Adam/bias/m_8Adam/kernel/vAdam/bias/vAdam/kernel/v_1Adam/bias/v_1Adam/kernel/v_2Adam/bias/v_2Adam/kernel/v_3Adam/bias/v_3Adam/kernel/v_4Adam/bias/v_4Adam/kernel/v_5Adam/bias/v_5Adam/kernel/v_6Adam/bias/v_6Adam/kernel/v_7Adam/bias/v_7Adam/kernel/v_8Adam/bias/v_8*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *,
f'R%
#__inference__traced_restore_1594696??
?
?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1592901

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_1592581

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
#__inference__traced_restore_1594696
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate
assignvariableop_5_kernel
assignvariableop_6_bias
assignvariableop_7_kernel_1
assignvariableop_8_bias_1
assignvariableop_9_kernel_2
assignvariableop_10_bias_2 
assignvariableop_11_kernel_3
assignvariableop_12_bias_3 
assignvariableop_13_kernel_4
assignvariableop_14_bias_4 
assignvariableop_15_kernel_5
assignvariableop_16_bias_5 
assignvariableop_17_kernel_6
assignvariableop_18_bias_6 
assignvariableop_19_kernel_7
assignvariableop_20_bias_7 
assignvariableop_21_kernel_8
assignvariableop_22_bias_8
assignvariableop_23_total
assignvariableop_24_count%
!assignvariableop_25_adam_kernel_m#
assignvariableop_26_adam_bias_m'
#assignvariableop_27_adam_kernel_m_1%
!assignvariableop_28_adam_bias_m_1'
#assignvariableop_29_adam_kernel_m_2%
!assignvariableop_30_adam_bias_m_2'
#assignvariableop_31_adam_kernel_m_3%
!assignvariableop_32_adam_bias_m_3'
#assignvariableop_33_adam_kernel_m_4%
!assignvariableop_34_adam_bias_m_4'
#assignvariableop_35_adam_kernel_m_5%
!assignvariableop_36_adam_bias_m_5'
#assignvariableop_37_adam_kernel_m_6%
!assignvariableop_38_adam_bias_m_6'
#assignvariableop_39_adam_kernel_m_7%
!assignvariableop_40_adam_bias_m_7'
#assignvariableop_41_adam_kernel_m_8%
!assignvariableop_42_adam_bias_m_8%
!assignvariableop_43_adam_kernel_v#
assignvariableop_44_adam_bias_v'
#assignvariableop_45_adam_kernel_v_1%
!assignvariableop_46_adam_bias_v_1'
#assignvariableop_47_adam_kernel_v_2%
!assignvariableop_48_adam_bias_v_2'
#assignvariableop_49_adam_kernel_v_3%
!assignvariableop_50_adam_bias_v_3'
#assignvariableop_51_adam_kernel_v_4%
!assignvariableop_52_adam_bias_v_4'
#assignvariableop_53_adam_kernel_v_5%
!assignvariableop_54_adam_bias_v_5'
#assignvariableop_55_adam_kernel_v_6%
!assignvariableop_56_adam_bias_v_6'
#assignvariableop_57_adam_kernel_v_7%
!assignvariableop_58_adam_bias_v_7'
#assignvariableop_59_adam_kernel_v_8%
!assignvariableop_60_adam_bias_v_8
identity_62??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*? 
value? B? >B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*?
value?B?>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*L
dtypesB
@2>	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_kernel_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_bias_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_kernel_2Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_bias_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_kernel_3Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_bias_3Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_kernel_4Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_bias_4Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_kernel_5Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_bias_5Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_kernel_6Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_bias_6Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_kernel_7Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_bias_7Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_kernel_8Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_bias_8Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_adam_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp#assignvariableop_27_adam_kernel_m_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp!assignvariableop_28_adam_bias_m_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp#assignvariableop_29_adam_kernel_m_2Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp!assignvariableop_30_adam_bias_m_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp#assignvariableop_31_adam_kernel_m_3Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp!assignvariableop_32_adam_bias_m_3Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp#assignvariableop_33_adam_kernel_m_4Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp!assignvariableop_34_adam_bias_m_4Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp#assignvariableop_35_adam_kernel_m_5Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp!assignvariableop_36_adam_bias_m_5Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp#assignvariableop_37_adam_kernel_m_6Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp!assignvariableop_38_adam_bias_m_6Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp#assignvariableop_39_adam_kernel_m_7Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp!assignvariableop_40_adam_bias_m_7Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp#assignvariableop_41_adam_kernel_m_8Identity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp!assignvariableop_42_adam_bias_m_8Identity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp!assignvariableop_43_adam_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_adam_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp#assignvariableop_45_adam_kernel_v_1Identity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp!assignvariableop_46_adam_bias_v_1Identity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp#assignvariableop_47_adam_kernel_v_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp!assignvariableop_48_adam_bias_v_2Identity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp#assignvariableop_49_adam_kernel_v_3Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp!assignvariableop_50_adam_bias_v_3Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp#assignvariableop_51_adam_kernel_v_4Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp!assignvariableop_52_adam_bias_v_4Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp#assignvariableop_53_adam_kernel_v_5Identity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp!assignvariableop_54_adam_bias_v_5Identity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp#assignvariableop_55_adam_kernel_v_6Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp!assignvariableop_56_adam_bias_v_6Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp#assignvariableop_57_adam_kernel_v_7Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp!assignvariableop_58_adam_bias_v_7Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp#assignvariableop_59_adam_kernel_v_8Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp!assignvariableop_60_adam_bias_v_8Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_609
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_61Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_61?
Identity_62IdentityIdentity_61:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_62"#
identity_62Identity_62:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

*__inference_conv2d_2_layer_call_fn_1594149

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15925402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_2_layer_call_fn_1592469

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_15924632
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?#
?
D__inference_encoder_layer_call_and_return_conditional_losses_1592706

inputs
conv2d_1592681
conv2d_1592683
conv2d_1_1592687
conv2d_1_1592689
conv2d_2_1592693
conv2d_2_1592695
dense_1592700
dense_1592702
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1592681conv2d_1592683*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15924842 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_15924392
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1592687conv2d_1_1592689*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15925122"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15924512!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1592693conv2d_2_1592695*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15925402"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_15924632!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15925632
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1592700dense_1592702*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15925812
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
E
)__inference_flatten_layer_call_fn_1594160

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15925632
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?

*__inference_conv2d_6_layer_call_fn_1594297

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15929292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
h
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1592757

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1592845

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_1593895

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15927062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?*
?
D__inference_decoder_layer_call_and_return_conditional_losses_1593015

inputs
dense_1_1592985
dense_1_1592987
conv2d_3_1592991
conv2d_3_1592993
conv2d_4_1592997
conv2d_4_1592999
conv2d_5_1593003
conv2d_5_1593005
conv2d_6_1593009
conv2d_6_1593011
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_1592985dense_1_1592987*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_15927962!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_15928262
reshape/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_3_1592991conv2d_3_1592993*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15928452"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_15927382
up_sampling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_4_1592997conv2d_4_1592999*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15928732"
 conv2d_4/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_15927572!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_1593003conv2d_5_1593005*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15929012"
 conv2d_5/StatefulPartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_15927762!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_1593009conv2d_6_1593011*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15929292"
 conv2d_6/StatefulPartitionedCall?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_1593440
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_15934012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?

*__inference_conv2d_4_layer_call_fn_1594257

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15928732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_1592676
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15926572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?-
?
D__inference_encoder_layer_call_and_return_conditional_losses_1593853

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd}
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_1592563

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
f
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1592439

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
M
1__inference_max_pooling2d_1_layer_call_fn_1592457

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15924512
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?#
?
D__inference_encoder_layer_call_and_return_conditional_losses_1592657

inputs
conv2d_1592632
conv2d_1592634
conv2d_1_1592638
conv2d_1_1592640
conv2d_2_1592644
conv2d_2_1592646
dense_1592651
dense_1592653
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1592632conv2d_1592634*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15924842 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_15924392
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1592638conv2d_1_1592640*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15925122"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15924512!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1592644conv2d_2_1592646*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15925402"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_15924632!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15925632
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1592651dense_1592653*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15925812
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?-
?
D__inference_encoder_layer_call_and_return_conditional_losses_1593817

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAddw
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d/Relu?
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d/MaxPool?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_1/BiasAdd}
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_1/Relu?
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2
max_pooling2d_1/MaxPool?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/BiasAdd{
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_2/Relu?
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2
max_pooling2d_2/MaxPoolo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape max_pooling2d_2/MaxPool:output:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1594268

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1592512

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_1593038
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
M
1__inference_up_sampling2d_1_layer_call_fn_1592763

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_15927572
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_1593740

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_15933182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?*
?
D__inference_decoder_layer_call_and_return_conditional_losses_1592979
input_2
dense_1_1592949
dense_1_1592951
conv2d_3_1592955
conv2d_3_1592957
conv2d_4_1592961
conv2d_4_1592963
conv2d_5_1592967
conv2d_5_1592969
conv2d_6_1592973
conv2d_6_1592975
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_1_1592949dense_1_1592951*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_15927962!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_15928262
reshape/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_3_1592955conv2d_3_1592957*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15928452"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_15927382
up_sampling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_4_1592961conv2d_4_1592963*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15928732"
 conv2d_4/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_15927572!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_1592967conv2d_5_1592969*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15929012"
 conv2d_5/StatefulPartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_15927762!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_1592973conv2d_6_1592975*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15929292"
 conv2d_6/StatefulPartitionedCall?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1594140

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?_
?
D__inference_decoder_layer_call_and_return_conditional_losses_1593967

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape/Reshape?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dreshape/Reshape:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/Reluu
up_sampling2d/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_4/Reluy
up_sampling2d_1/ShapeShapeconv2d_4/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_4/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_5/BiasAdd}
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_5/Reluy
up_sampling2d_2/ShapeShapeconv2d_5/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape?
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack?
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1?
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2?
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const?
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_5/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_6/BiasAdd?
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_6/Sigmoid?
IdentityIdentityconv2d_6/Sigmoid:y:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
|
'__inference_dense_layer_call_fn_1594179

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15925812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1592873

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?s
?
 __inference__traced_save_1594503
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop%
!savev2_kernel_read_readvariableop#
savev2_bias_read_readvariableop'
#savev2_kernel_1_read_readvariableop%
!savev2_bias_1_read_readvariableop'
#savev2_kernel_2_read_readvariableop%
!savev2_bias_2_read_readvariableop'
#savev2_kernel_3_read_readvariableop%
!savev2_bias_3_read_readvariableop'
#savev2_kernel_4_read_readvariableop%
!savev2_bias_4_read_readvariableop'
#savev2_kernel_5_read_readvariableop%
!savev2_bias_5_read_readvariableop'
#savev2_kernel_6_read_readvariableop%
!savev2_bias_6_read_readvariableop'
#savev2_kernel_7_read_readvariableop%
!savev2_bias_7_read_readvariableop'
#savev2_kernel_8_read_readvariableop%
!savev2_bias_8_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop,
(savev2_adam_kernel_m_read_readvariableop*
&savev2_adam_bias_m_read_readvariableop.
*savev2_adam_kernel_m_1_read_readvariableop,
(savev2_adam_bias_m_1_read_readvariableop.
*savev2_adam_kernel_m_2_read_readvariableop,
(savev2_adam_bias_m_2_read_readvariableop.
*savev2_adam_kernel_m_3_read_readvariableop,
(savev2_adam_bias_m_3_read_readvariableop.
*savev2_adam_kernel_m_4_read_readvariableop,
(savev2_adam_bias_m_4_read_readvariableop.
*savev2_adam_kernel_m_5_read_readvariableop,
(savev2_adam_bias_m_5_read_readvariableop.
*savev2_adam_kernel_m_6_read_readvariableop,
(savev2_adam_bias_m_6_read_readvariableop.
*savev2_adam_kernel_m_7_read_readvariableop,
(savev2_adam_bias_m_7_read_readvariableop.
*savev2_adam_kernel_m_8_read_readvariableop,
(savev2_adam_bias_m_8_read_readvariableop,
(savev2_adam_kernel_v_read_readvariableop*
&savev2_adam_bias_v_read_readvariableop.
*savev2_adam_kernel_v_1_read_readvariableop,
(savev2_adam_bias_v_1_read_readvariableop.
*savev2_adam_kernel_v_2_read_readvariableop,
(savev2_adam_bias_v_2_read_readvariableop.
*savev2_adam_kernel_v_3_read_readvariableop,
(savev2_adam_bias_v_3_read_readvariableop.
*savev2_adam_kernel_v_4_read_readvariableop,
(savev2_adam_bias_v_4_read_readvariableop.
*savev2_adam_kernel_v_5_read_readvariableop,
(savev2_adam_bias_v_5_read_readvariableop.
*savev2_adam_kernel_v_6_read_readvariableop,
(savev2_adam_bias_v_6_read_readvariableop.
*savev2_adam_kernel_v_7_read_readvariableop,
(savev2_adam_bias_v_7_read_readvariableop.
*savev2_adam_kernel_v_8_read_readvariableop,
(savev2_adam_bias_v_8_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*? 
value? B? >B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:>*
dtype0*?
value?B?>B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop!savev2_kernel_read_readvariableopsavev2_bias_read_readvariableop#savev2_kernel_1_read_readvariableop!savev2_bias_1_read_readvariableop#savev2_kernel_2_read_readvariableop!savev2_bias_2_read_readvariableop#savev2_kernel_3_read_readvariableop!savev2_bias_3_read_readvariableop#savev2_kernel_4_read_readvariableop!savev2_bias_4_read_readvariableop#savev2_kernel_5_read_readvariableop!savev2_bias_5_read_readvariableop#savev2_kernel_6_read_readvariableop!savev2_bias_6_read_readvariableop#savev2_kernel_7_read_readvariableop!savev2_bias_7_read_readvariableop#savev2_kernel_8_read_readvariableop!savev2_bias_8_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop(savev2_adam_kernel_m_read_readvariableop&savev2_adam_bias_m_read_readvariableop*savev2_adam_kernel_m_1_read_readvariableop(savev2_adam_bias_m_1_read_readvariableop*savev2_adam_kernel_m_2_read_readvariableop(savev2_adam_bias_m_2_read_readvariableop*savev2_adam_kernel_m_3_read_readvariableop(savev2_adam_bias_m_3_read_readvariableop*savev2_adam_kernel_m_4_read_readvariableop(savev2_adam_bias_m_4_read_readvariableop*savev2_adam_kernel_m_5_read_readvariableop(savev2_adam_bias_m_5_read_readvariableop*savev2_adam_kernel_m_6_read_readvariableop(savev2_adam_bias_m_6_read_readvariableop*savev2_adam_kernel_m_7_read_readvariableop(savev2_adam_bias_m_7_read_readvariableop*savev2_adam_kernel_m_8_read_readvariableop(savev2_adam_bias_m_8_read_readvariableop(savev2_adam_kernel_v_read_readvariableop&savev2_adam_bias_v_read_readvariableop*savev2_adam_kernel_v_1_read_readvariableop(savev2_adam_bias_v_1_read_readvariableop*savev2_adam_kernel_v_2_read_readvariableop(savev2_adam_bias_v_2_read_readvariableop*savev2_adam_kernel_v_3_read_readvariableop(savev2_adam_bias_v_3_read_readvariableop*savev2_adam_kernel_v_4_read_readvariableop(savev2_adam_bias_v_4_read_readvariableop*savev2_adam_kernel_v_5_read_readvariableop(savev2_adam_bias_v_5_read_readvariableop*savev2_adam_kernel_v_6_read_readvariableop(savev2_adam_bias_v_6_read_readvariableop*savev2_adam_kernel_v_7_read_readvariableop(savev2_adam_bias_v_7_read_readvariableop*savev2_adam_kernel_v_8_read_readvariableop(savev2_adam_bias_v_8_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *L
dtypesB
@2>	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : ::::: : :	?::	?:?:  : : :::::: : ::::: : :	?::	?:?:  : : :::::::::: : :	?::	?:?:  : : :::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 	

_output_shapes
::,
(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :% !

_output_shapes
:	?: !

_output_shapes
::%"!

_output_shapes
:	?:!#

_output_shapes	
:?:,$(
&
_output_shapes
:  : %

_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
::,((
&
_output_shapes
:: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
: : 1

_output_shapes
: :%2!

_output_shapes
:	?: 3

_output_shapes
::%4!

_output_shapes
:	?:!5

_output_shapes	
:?:,6(
&
_output_shapes
:  : 7

_output_shapes
: :,8(
&
_output_shapes
: : 9

_output_shapes
::,:(
&
_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::>

_output_shapes
: 
?
?
)__inference_encoder_layer_call_fn_1593874

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15926572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1592451

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1594248

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_1594155

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?*
?
D__inference_decoder_layer_call_and_return_conditional_losses_1593073

inputs
dense_1_1593043
dense_1_1593045
conv2d_3_1593049
conv2d_3_1593051
conv2d_4_1593055
conv2d_4_1593057
conv2d_5_1593061
conv2d_5_1593063
conv2d_6_1593067
conv2d_6_1593069
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_1593043dense_1_1593045*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_15927962!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_15928262
reshape/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_3_1593049conv2d_3_1593051*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15928452"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_15927382
up_sampling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_4_1593055conv2d_4_1593057*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15928732"
 conv2d_4/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_15927572!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_1593061conv2d_5_1593063*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15929012"
 conv2d_5/StatefulPartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_15927762!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_1593067conv2d_6_1593069*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15929292"
 conv2d_6/StatefulPartitionedCall?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1592776

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593318

inputs
encoder_1593279
encoder_1593281
encoder_1593283
encoder_1593285
encoder_1593287
encoder_1593289
encoder_1593291
encoder_1593293
decoder_1593296
decoder_1593298
decoder_1593300
decoder_1593302
decoder_1593304
decoder_1593306
decoder_1593308
decoder_1593310
decoder_1593312
decoder_1593314
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_1593279encoder_1593281encoder_1593283encoder_1593285encoder_1593287encoder_1593289encoder_1593291encoder_1593293*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15926572!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_1593296decoder_1593298decoder_1593300decoder_1593302decoder_1593304decoder_1593306decoder_1593308decoder_1593310decoder_1593312decoder_1593314*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930152!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593231
input_1
encoder_1593142
encoder_1593144
encoder_1593146
encoder_1593148
encoder_1593150
encoder_1593152
encoder_1593154
encoder_1593156
decoder_1593209
decoder_1593211
decoder_1593213
decoder_1593215
decoder_1593217
decoder_1593219
decoder_1593221
decoder_1593223
decoder_1593225
decoder_1593227
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_1593142encoder_1593144encoder_1593146encoder_1593148encoder_1593150encoder_1593152encoder_1593154encoder_1593156*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15926572!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_1593209decoder_1593211decoder_1593213decoder_1593215decoder_1593217decoder_1593219decoder_1593221decoder_1593223decoder_1593225decoder_1593227*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930152!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
-__inference_autoencoder_layer_call_fn_1593357
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_15933182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?#
?
D__inference_encoder_layer_call_and_return_conditional_losses_1592598
input_1
conv2d_1592495
conv2d_1592497
conv2d_1_1592523
conv2d_1_1592525
conv2d_2_1592551
conv2d_2_1592553
dense_1592592
dense_1592594
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1592495conv2d_1592497*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15924842 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_15924392
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1592523conv2d_1_1592525*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15925122"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15924512!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1592551conv2d_2_1592553*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15925402"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_15924632!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15925632
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1592592dense_1592594*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15925812
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_1592796

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
h
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1592463

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

*__inference_conv2d_3_layer_call_fn_1594237

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15928452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
~
)__inference_dense_1_layer_call_fn_1594198

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_15927962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
/__inference_max_pooling2d_layer_call_fn_1592445

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_15924392
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?_
?
D__inference_decoder_layer_call_and_return_conditional_losses_1594039

inputs*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource+
'conv2d_3_conv2d_readvariableop_resource,
(conv2d_3_biasadd_readvariableop_resource+
'conv2d_4_conv2d_readvariableop_resource,
(conv2d_4_biasadd_readvariableop_resource+
'conv2d_5_conv2d_readvariableop_resource,
(conv2d_5_biasadd_readvariableop_resource+
'conv2d_6_conv2d_readvariableop_resource,
(conv2d_6_biasadd_readvariableop_resource
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_1/BiasAddf
reshape/ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape/Reshape?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dreshape/Reshape:output:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd{
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/Reluu
up_sampling2d/ShapeShapeconv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d/Shape?
!up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2#
!up_sampling2d/strided_slice/stack?
#up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_1?
#up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d/strided_slice/stack_2?
up_sampling2d/strided_sliceStridedSliceup_sampling2d/Shape:output:0*up_sampling2d/strided_slice/stack:output:0,up_sampling2d/strided_slice/stack_1:output:0,up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d/strided_slice{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const?
up_sampling2d/mulMul$up_sampling2d/strided_slice:output:0up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_3/Relu:activations:0up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_4/BiasAdd{
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_4/Reluy
up_sampling2d_1/ShapeShapeconv2d_4/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_1/Shape?
#up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_1/strided_slice/stack?
%up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_1?
%up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_1/strided_slice/stack_2?
up_sampling2d_1/strided_sliceStridedSliceup_sampling2d_1/Shape:output:0,up_sampling2d_1/strided_slice/stack:output:0.up_sampling2d_1/strided_slice/stack_1:output:0.up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_1/strided_slice
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/mulMul&up_sampling2d_1/strided_slice:output:0up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_4/Relu:activations:0up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_5/BiasAdd}
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_5/Reluy
up_sampling2d_2/ShapeShapeconv2d_5/Relu:activations:0*
T0*
_output_shapes
:2
up_sampling2d_2/Shape?
#up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_2/strided_slice/stack?
%up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_1?
%up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_2/strided_slice/stack_2?
up_sampling2d_2/strided_sliceStridedSliceup_sampling2d_2/Shape:output:0,up_sampling2d_2/strided_slice/stack:output:0.up_sampling2d_2/strided_slice/stack_1:output:0.up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_2/strided_slice
up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_2/Const?
up_sampling2d_2/mulMul&up_sampling2d_2/strided_slice:output:0up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_2/mul?
,up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighborconv2d_5/Relu:activations:0up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2.
,up_sampling2d_2/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d_6/BiasAdd?
conv2d_6/SigmoidSigmoidconv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
conv2d_6/Sigmoid?
IdentityIdentityconv2d_6/Sigmoid:y:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?*
?
D__inference_decoder_layer_call_and_return_conditional_losses_1592946
input_2
dense_1_1592807
dense_1_1592809
conv2d_3_1592856
conv2d_3_1592858
conv2d_4_1592884
conv2d_4_1592886
conv2d_5_1592912
conv2d_5_1592914
conv2d_6_1592940
conv2d_6_1592942
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall? conv2d_6/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_1_1592807dense_1_1592809*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_15927962!
dense_1/StatefulPartitionedCall?
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_15928262
reshape/PartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv2d_3_1592856conv2d_3_1592858*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_3_layer_call_and_return_conditional_losses_15928452"
 conv2d_3/StatefulPartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_15927382
up_sampling2d/PartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_4_1592884conv2d_4_1592886*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_4_layer_call_and_return_conditional_losses_15928732"
 conv2d_4/StatefulPartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_15927572!
up_sampling2d_1/PartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_5_1592912conv2d_5_1592914*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15929012"
 conv2d_5/StatefulPartitionedCall?
up_sampling2d_2/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_15927762!
up_sampling2d_2/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_2/PartitionedCall:output:0conv2d_6_1592940conv2d_6_1592942*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_6_layer_call_and_return_conditional_losses_15929292"
 conv2d_6/StatefulPartitionedCall?
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_1592826

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:????????? 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
}
(__inference_conv2d_layer_call_fn_1594109

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15924842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_1594089

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?#
?
D__inference_encoder_layer_call_and_return_conditional_losses_1592626
input_1
conv2d_1592601
conv2d_1592603
conv2d_1_1592607
conv2d_1_1592609
conv2d_2_1592613
conv2d_2_1592615
dense_1592620
dense_1592622
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_1592601conv2d_1592603*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *L
fGRE
C__inference_conv2d_layer_call_and_return_conditional_losses_15924842 
conv2d/StatefulPartitionedCall?
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_15924392
max_pooling2d/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_1592607conv2d_1_1592609*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15925122"
 conv2d_1/StatefulPartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_15924512!
max_pooling2d_1/PartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_1592613conv2d_2_1592615*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_2_layer_call_and_return_conditional_losses_15925402"
 conv2d_2/StatefulPartitionedCall?
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_15924632!
max_pooling2d_2/PartitionedCall?
flatten/PartitionedCallPartitionedCall(max_pooling2d_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_15925632
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_1592620dense_1592622*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_15925812
dense/StatefulPartitionedCall?
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
%__inference_signature_wrapper_1593491
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*4
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *+
f&R$
"__inference__wrapped_model_15924332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593401

inputs
encoder_1593362
encoder_1593364
encoder_1593366
encoder_1593368
encoder_1593370
encoder_1593372
encoder_1593374
encoder_1593376
decoder_1593379
decoder_1593381
decoder_1593383
decoder_1593385
decoder_1593387
decoder_1593389
decoder_1593391
decoder_1593393
decoder_1593395
decoder_1593397
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_1593362encoder_1593364encoder_1593366encoder_1593368encoder_1593370encoder_1593372encoder_1593374encoder_1593376*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15927062!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_1593379decoder_1593381decoder_1593383decoder_1593385decoder_1593387decoder_1593389decoder_1593391decoder_1593393decoder_1593395decoder_1593397*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930732!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
-__inference_autoencoder_layer_call_fn_1593781

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*4
_read_only_resource_inputs
	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *Q
fLRJ
H__inference_autoencoder_layer_call_and_return_conditional_losses_15934012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

*__inference_conv2d_1_layer_call_fn_1594129

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_1_layer_call_and_return_conditional_losses_15925122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
K
/__inference_up_sampling2d_layer_call_fn_1592744

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *S
fNRL
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_15927382
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1592540

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_1593096
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?	
?
B__inference_dense_layer_call_and_return_conditional_losses_1594170

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*/
_input_shapes
:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
)__inference_decoder_layer_call_fn_1594064

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*N
_input_shapes=
;:?????????::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
f
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1592738

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593273
input_1
encoder_1593234
encoder_1593236
encoder_1593238
encoder_1593240
encoder_1593242
encoder_1593244
encoder_1593246
encoder_1593248
decoder_1593251
decoder_1593253
decoder_1593255
decoder_1593257
decoder_1593259
decoder_1593261
decoder_1593263
decoder_1593265
decoder_1593267
decoder_1593269
identity??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_1593234encoder_1593236encoder_1593238encoder_1593240encoder_1593242encoder_1593244encoder_1593246encoder_1593248*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15927062!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:0decoder_1593251decoder_1593253decoder_1593255decoder_1593257decoder_1593259decoder_1593261decoder_1593263decoder_1593265decoder_1593267decoder_1593269*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*,
_read_only_resource_inputs

	
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_decoder_layer_call_and_return_conditional_losses_15930732!
decoder/StatefulPartitionedCall?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
E
)__inference_reshape_layer_call_fn_1594217

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_15928262
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_1592484

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593699

inputs1
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource3
/encoder_conv2d_1_conv2d_readvariableop_resource4
0encoder_conv2d_1_biasadd_readvariableop_resource3
/encoder_conv2d_2_conv2d_readvariableop_resource4
0encoder_conv2d_2_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.decoder_dense_1_matmul_readvariableop_resource3
/decoder_dense_1_biasadd_readvariableop_resource3
/decoder_conv2d_3_conv2d_readvariableop_resource4
0decoder_conv2d_3_biasadd_readvariableop_resource3
/decoder_conv2d_4_conv2d_readvariableop_resource4
0decoder_conv2d_4_biasadd_readvariableop_resource3
/decoder_conv2d_5_conv2d_readvariableop_resource4
0decoder_conv2d_5_biasadd_readvariableop_resource3
/decoder_conv2d_6_conv2d_readvariableop_resource4
0decoder_conv2d_6_biasadd_readvariableop_resource
identity??'decoder/conv2d_3/BiasAdd/ReadVariableOp?&decoder/conv2d_3/Conv2D/ReadVariableOp?'decoder/conv2d_4/BiasAdd/ReadVariableOp?&decoder/conv2d_4/Conv2D/ReadVariableOp?'decoder/conv2d_5/BiasAdd/ReadVariableOp?&decoder/conv2d_5/Conv2D/ReadVariableOp?'decoder/conv2d_6/BiasAdd/ReadVariableOp?&decoder/conv2d_6/Conv2D/ReadVariableOp?&decoder/dense_1/BiasAdd/ReadVariableOp?%decoder/dense_1/MatMul/ReadVariableOp?%encoder/conv2d/BiasAdd/ReadVariableOp?$encoder/conv2d/Conv2D/ReadVariableOp?'encoder/conv2d_1/BiasAdd/ReadVariableOp?&encoder/conv2d_1/Conv2D/ReadVariableOp?'encoder/conv2d_2/BiasAdd/ReadVariableOp?&encoder/conv2d_2/Conv2D/ReadVariableOp?$encoder/dense/BiasAdd/ReadVariableOp?#encoder/dense/MatMul/ReadVariableOp?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d/BiasAdd?
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d/Relu?
encoder/max_pooling2d/MaxPoolMaxPool!encoder/conv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
encoder/max_pooling2d/MaxPool?
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&encoder/conv2d_1/Conv2D/ReadVariableOp?
encoder/conv2d_1/Conv2DConv2D&encoder/max_pooling2d/MaxPool:output:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
encoder/conv2d_1/Conv2D?
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'encoder/conv2d_1/BiasAdd/ReadVariableOp?
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d_1/BiasAdd?
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d_1/Relu?
encoder/max_pooling2d_1/MaxPoolMaxPool#encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling2d_1/MaxPool?
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&encoder/conv2d_2/Conv2D/ReadVariableOp?
encoder/conv2d_2/Conv2DConv2D(encoder/max_pooling2d_1/MaxPool:output:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
encoder/conv2d_2/Conv2D?
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv2d_2/BiasAdd/ReadVariableOp?
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_2/BiasAdd?
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_2/Relu?
encoder/max_pooling2d_2/MaxPoolMaxPool#encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling2d_2/MaxPool
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape(encoder/max_pooling2d_2/MaxPool:output:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/BiasAdd?
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%decoder/dense_1/MatMul/ReadVariableOp?
decoder/dense_1/MatMulMatMulencoder/dense/BiasAdd:output:0-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_1/MatMul?
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_1/BiasAdd/ReadVariableOp?
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_1/BiasAdd~
decoder/reshape/ShapeShape decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2!
decoder/reshape/Reshape/shape/3?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape decoder/dense_1/BiasAdd:output:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
decoder/reshape/Reshape?
&decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02(
&decoder/conv2d_3/Conv2D/ReadVariableOp?
decoder/conv2d_3/Conv2DConv2D decoder/reshape/Reshape:output:0.decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
decoder/conv2d_3/Conv2D?
'decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'decoder/conv2d_3/BiasAdd/ReadVariableOp?
decoder/conv2d_3/BiasAddBiasAdd decoder/conv2d_3/Conv2D:output:0/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
decoder/conv2d_3/BiasAdd?
decoder/conv2d_3/ReluRelu!decoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
decoder/conv2d_3/Relu?
decoder/up_sampling2d/ShapeShape#decoder/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
decoder/up_sampling2d/Shape?
)decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)decoder/up_sampling2d/strided_slice/stack?
+decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d/strided_slice/stack_1?
+decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d/strided_slice/stack_2?
#decoder/up_sampling2d/strided_sliceStridedSlice$decoder/up_sampling2d/Shape:output:02decoder/up_sampling2d/strided_slice/stack:output:04decoder/up_sampling2d/strided_slice/stack_1:output:04decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#decoder/up_sampling2d/strided_slice?
decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
decoder/up_sampling2d/Const?
decoder/up_sampling2d/mulMul,decoder/up_sampling2d/strided_slice:output:0$decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
decoder/up_sampling2d/mul?
2decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_3/Relu:activations:0decoder/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(24
2decoder/up_sampling2d/resize/ResizeNearestNeighbor?
&decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&decoder/conv2d_4/Conv2D/ReadVariableOp?
decoder/conv2d_4/Conv2DConv2DCdecoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
decoder/conv2d_4/Conv2D?
'decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_4/BiasAdd/ReadVariableOp?
decoder/conv2d_4/BiasAddBiasAdd decoder/conv2d_4/Conv2D:output:0/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder/conv2d_4/BiasAdd?
decoder/conv2d_4/ReluRelu!decoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder/conv2d_4/Relu?
decoder/up_sampling2d_1/ShapeShape#decoder/conv2d_4/Relu:activations:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_1/Shape?
+decoder/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d_1/strided_slice/stack?
-decoder/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_1/strided_slice/stack_1?
-decoder/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_1/strided_slice/stack_2?
%decoder/up_sampling2d_1/strided_sliceStridedSlice&decoder/up_sampling2d_1/Shape:output:04decoder/up_sampling2d_1/strided_slice/stack:output:06decoder/up_sampling2d_1/strided_slice/stack_1:output:06decoder/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%decoder/up_sampling2d_1/strided_slice?
decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
decoder/up_sampling2d_1/Const?
decoder/up_sampling2d_1/mulMul.decoder/up_sampling2d_1/strided_slice:output:0&decoder/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_1/mul?
4decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_4/Relu:activations:0decoder/up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(26
4decoder/up_sampling2d_1/resize/ResizeNearestNeighbor?
&decoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&decoder/conv2d_5/Conv2D/ReadVariableOp?
decoder/conv2d_5/Conv2DConv2DEdecoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
decoder/conv2d_5/Conv2D?
'decoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_5/BiasAdd/ReadVariableOp?
decoder/conv2d_5/BiasAddBiasAdd decoder/conv2d_5/Conv2D:output:0/decoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_5/BiasAdd?
decoder/conv2d_5/ReluRelu!decoder/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_5/Relu?
decoder/up_sampling2d_2/ShapeShape#decoder/conv2d_5/Relu:activations:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_2/Shape?
+decoder/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d_2/strided_slice/stack?
-decoder/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_2/strided_slice/stack_1?
-decoder/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_2/strided_slice/stack_2?
%decoder/up_sampling2d_2/strided_sliceStridedSlice&decoder/up_sampling2d_2/Shape:output:04decoder/up_sampling2d_2/strided_slice/stack:output:06decoder/up_sampling2d_2/strided_slice/stack_1:output:06decoder/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%decoder/up_sampling2d_2/strided_slice?
decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
decoder/up_sampling2d_2/Const?
decoder/up_sampling2d_2/mulMul.decoder/up_sampling2d_2/strided_slice:output:0&decoder/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_2/mul?
4decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_5/Relu:activations:0decoder/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(26
4decoder/up_sampling2d_2/resize/ResizeNearestNeighbor?
&decoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&decoder/conv2d_6/Conv2D/ReadVariableOp?
decoder/conv2d_6/Conv2DConv2DEdecoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
decoder/conv2d_6/Conv2D?
'decoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_6/BiasAdd/ReadVariableOp?
decoder/conv2d_6/BiasAddBiasAdd decoder/conv2d_6/Conv2D:output:0/decoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_6/BiasAdd?
decoder/conv2d_6/SigmoidSigmoid!decoder/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_6/Sigmoid?
IdentityIdentitydecoder/conv2d_6/Sigmoid:y:0(^decoder/conv2d_3/BiasAdd/ReadVariableOp'^decoder/conv2d_3/Conv2D/ReadVariableOp(^decoder/conv2d_4/BiasAdd/ReadVariableOp'^decoder/conv2d_4/Conv2D/ReadVariableOp(^decoder/conv2d_5/BiasAdd/ReadVariableOp'^decoder/conv2d_5/Conv2D/ReadVariableOp(^decoder/conv2d_6/BiasAdd/ReadVariableOp'^decoder/conv2d_6/Conv2D/ReadVariableOp'^decoder/dense_1/BiasAdd/ReadVariableOp&^decoder/dense_1/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::2R
'decoder/conv2d_3/BiasAdd/ReadVariableOp'decoder/conv2d_3/BiasAdd/ReadVariableOp2P
&decoder/conv2d_3/Conv2D/ReadVariableOp&decoder/conv2d_3/Conv2D/ReadVariableOp2R
'decoder/conv2d_4/BiasAdd/ReadVariableOp'decoder/conv2d_4/BiasAdd/ReadVariableOp2P
&decoder/conv2d_4/Conv2D/ReadVariableOp&decoder/conv2d_4/Conv2D/ReadVariableOp2R
'decoder/conv2d_5/BiasAdd/ReadVariableOp'decoder/conv2d_5/BiasAdd/ReadVariableOp2P
&decoder/conv2d_5/Conv2D/ReadVariableOp&decoder/conv2d_5/Conv2D/ReadVariableOp2R
'decoder/conv2d_6/BiasAdd/ReadVariableOp'decoder/conv2d_6/BiasAdd/ReadVariableOp2P
&decoder/conv2d_6/Conv2D/ReadVariableOp&decoder/conv2d_6/Conv2D/ReadVariableOp2P
&decoder/dense_1/BiasAdd/ReadVariableOp&decoder/dense_1/BiasAdd/ReadVariableOp2N
%decoder/dense_1/MatMul/ReadVariableOp%decoder/dense_1/MatMul/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1594228

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
D__inference_dense_1_layer_call_and_return_conditional_losses_1594189

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_encoder_layer_call_fn_1592725
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????**
_read_only_resource_inputs

*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_15927062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:???????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?

?
C__inference_conv2d_layer_call_and_return_conditional_losses_1594100

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593595

inputs1
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource3
/encoder_conv2d_1_conv2d_readvariableop_resource4
0encoder_conv2d_1_biasadd_readvariableop_resource3
/encoder_conv2d_2_conv2d_readvariableop_resource4
0encoder_conv2d_2_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.decoder_dense_1_matmul_readvariableop_resource3
/decoder_dense_1_biasadd_readvariableop_resource3
/decoder_conv2d_3_conv2d_readvariableop_resource4
0decoder_conv2d_3_biasadd_readvariableop_resource3
/decoder_conv2d_4_conv2d_readvariableop_resource4
0decoder_conv2d_4_biasadd_readvariableop_resource3
/decoder_conv2d_5_conv2d_readvariableop_resource4
0decoder_conv2d_5_biasadd_readvariableop_resource3
/decoder_conv2d_6_conv2d_readvariableop_resource4
0decoder_conv2d_6_biasadd_readvariableop_resource
identity??'decoder/conv2d_3/BiasAdd/ReadVariableOp?&decoder/conv2d_3/Conv2D/ReadVariableOp?'decoder/conv2d_4/BiasAdd/ReadVariableOp?&decoder/conv2d_4/Conv2D/ReadVariableOp?'decoder/conv2d_5/BiasAdd/ReadVariableOp?&decoder/conv2d_5/Conv2D/ReadVariableOp?'decoder/conv2d_6/BiasAdd/ReadVariableOp?&decoder/conv2d_6/Conv2D/ReadVariableOp?&decoder/dense_1/BiasAdd/ReadVariableOp?%decoder/dense_1/MatMul/ReadVariableOp?%encoder/conv2d/BiasAdd/ReadVariableOp?$encoder/conv2d/Conv2D/ReadVariableOp?'encoder/conv2d_1/BiasAdd/ReadVariableOp?&encoder/conv2d_1/Conv2D/ReadVariableOp?'encoder/conv2d_2/BiasAdd/ReadVariableOp?&encoder/conv2d_2/Conv2D/ReadVariableOp?$encoder/dense/BiasAdd/ReadVariableOp?#encoder/dense/MatMul/ReadVariableOp?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d/BiasAdd?
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d/Relu?
encoder/max_pooling2d/MaxPoolMaxPool!encoder/conv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
encoder/max_pooling2d/MaxPool?
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&encoder/conv2d_1/Conv2D/ReadVariableOp?
encoder/conv2d_1/Conv2DConv2D&encoder/max_pooling2d/MaxPool:output:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
encoder/conv2d_1/Conv2D?
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'encoder/conv2d_1/BiasAdd/ReadVariableOp?
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d_1/BiasAdd?
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
encoder/conv2d_1/Relu?
encoder/max_pooling2d_1/MaxPoolMaxPool#encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling2d_1/MaxPool?
&encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&encoder/conv2d_2/Conv2D/ReadVariableOp?
encoder/conv2d_2/Conv2DConv2D(encoder/max_pooling2d_1/MaxPool:output:0.encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
encoder/conv2d_2/Conv2D?
'encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv2d_2/BiasAdd/ReadVariableOp?
encoder/conv2d_2/BiasAddBiasAdd encoder/conv2d_2/Conv2D:output:0/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_2/BiasAdd?
encoder/conv2d_2/ReluRelu!encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d_2/Relu?
encoder/max_pooling2d_2/MaxPoolMaxPool#encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2!
encoder/max_pooling2d_2/MaxPool
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape(encoder/max_pooling2d_2/MaxPool:output:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/BiasAdd?
%decoder/dense_1/MatMul/ReadVariableOpReadVariableOp.decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%decoder/dense_1/MatMul/ReadVariableOp?
decoder/dense_1/MatMulMatMulencoder/dense/BiasAdd:output:0-decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_1/MatMul?
&decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_1/BiasAdd/ReadVariableOp?
decoder/dense_1/BiasAddBiasAdd decoder/dense_1/MatMul:product:0.decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_1/BiasAdd~
decoder/reshape/ShapeShape decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2!
decoder/reshape/Reshape/shape/3?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape decoder/dense_1/BiasAdd:output:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
decoder/reshape/Reshape?
&decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02(
&decoder/conv2d_3/Conv2D/ReadVariableOp?
decoder/conv2d_3/Conv2DConv2D decoder/reshape/Reshape:output:0.decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
decoder/conv2d_3/Conv2D?
'decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'decoder/conv2d_3/BiasAdd/ReadVariableOp?
decoder/conv2d_3/BiasAddBiasAdd decoder/conv2d_3/Conv2D:output:0/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
decoder/conv2d_3/BiasAdd?
decoder/conv2d_3/ReluRelu!decoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
decoder/conv2d_3/Relu?
decoder/up_sampling2d/ShapeShape#decoder/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2
decoder/up_sampling2d/Shape?
)decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2+
)decoder/up_sampling2d/strided_slice/stack?
+decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d/strided_slice/stack_1?
+decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d/strided_slice/stack_2?
#decoder/up_sampling2d/strided_sliceStridedSlice$decoder/up_sampling2d/Shape:output:02decoder/up_sampling2d/strided_slice/stack:output:04decoder/up_sampling2d/strided_slice/stack_1:output:04decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2%
#decoder/up_sampling2d/strided_slice?
decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
decoder/up_sampling2d/Const?
decoder/up_sampling2d/mulMul,decoder/up_sampling2d/strided_slice:output:0$decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2
decoder/up_sampling2d/mul?
2decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_3/Relu:activations:0decoder/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(24
2decoder/up_sampling2d/resize/ResizeNearestNeighbor?
&decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02(
&decoder/conv2d_4/Conv2D/ReadVariableOp?
decoder/conv2d_4/Conv2DConv2DCdecoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
decoder/conv2d_4/Conv2D?
'decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_4/BiasAdd/ReadVariableOp?
decoder/conv2d_4/BiasAddBiasAdd decoder/conv2d_4/Conv2D:output:0/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
decoder/conv2d_4/BiasAdd?
decoder/conv2d_4/ReluRelu!decoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
decoder/conv2d_4/Relu?
decoder/up_sampling2d_1/ShapeShape#decoder/conv2d_4/Relu:activations:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_1/Shape?
+decoder/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d_1/strided_slice/stack?
-decoder/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_1/strided_slice/stack_1?
-decoder/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_1/strided_slice/stack_2?
%decoder/up_sampling2d_1/strided_sliceStridedSlice&decoder/up_sampling2d_1/Shape:output:04decoder/up_sampling2d_1/strided_slice/stack:output:06decoder/up_sampling2d_1/strided_slice/stack_1:output:06decoder/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%decoder/up_sampling2d_1/strided_slice?
decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
decoder/up_sampling2d_1/Const?
decoder/up_sampling2d_1/mulMul.decoder/up_sampling2d_1/strided_slice:output:0&decoder/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_1/mul?
4decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_4/Relu:activations:0decoder/up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(26
4decoder/up_sampling2d_1/resize/ResizeNearestNeighbor?
&decoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&decoder/conv2d_5/Conv2D/ReadVariableOp?
decoder/conv2d_5/Conv2DConv2DEdecoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
decoder/conv2d_5/Conv2D?
'decoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_5/BiasAdd/ReadVariableOp?
decoder/conv2d_5/BiasAddBiasAdd decoder/conv2d_5/Conv2D:output:0/decoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_5/BiasAdd?
decoder/conv2d_5/ReluRelu!decoder/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_5/Relu?
decoder/up_sampling2d_2/ShapeShape#decoder/conv2d_5/Relu:activations:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_2/Shape?
+decoder/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2-
+decoder/up_sampling2d_2/strided_slice/stack?
-decoder/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_2/strided_slice/stack_1?
-decoder/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-decoder/up_sampling2d_2/strided_slice/stack_2?
%decoder/up_sampling2d_2/strided_sliceStridedSlice&decoder/up_sampling2d_2/Shape:output:04decoder/up_sampling2d_2/strided_slice/stack:output:06decoder/up_sampling2d_2/strided_slice/stack_1:output:06decoder/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2'
%decoder/up_sampling2d_2/strided_slice?
decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
decoder/up_sampling2d_2/Const?
decoder/up_sampling2d_2/mulMul.decoder/up_sampling2d_2/strided_slice:output:0&decoder/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2
decoder/up_sampling2d_2/mul?
4decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor#decoder/conv2d_5/Relu:activations:0decoder/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(26
4decoder/up_sampling2d_2/resize/ResizeNearestNeighbor?
&decoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp/decoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02(
&decoder/conv2d_6/Conv2D/ReadVariableOp?
decoder/conv2d_6/Conv2DConv2DEdecoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0.decoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
decoder/conv2d_6/Conv2D?
'decoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp0decoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'decoder/conv2d_6/BiasAdd/ReadVariableOp?
decoder/conv2d_6/BiasAddBiasAdd decoder/conv2d_6/Conv2D:output:0/decoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_6/BiasAdd?
decoder/conv2d_6/SigmoidSigmoid!decoder/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
decoder/conv2d_6/Sigmoid?
IdentityIdentitydecoder/conv2d_6/Sigmoid:y:0(^decoder/conv2d_3/BiasAdd/ReadVariableOp'^decoder/conv2d_3/Conv2D/ReadVariableOp(^decoder/conv2d_4/BiasAdd/ReadVariableOp'^decoder/conv2d_4/Conv2D/ReadVariableOp(^decoder/conv2d_5/BiasAdd/ReadVariableOp'^decoder/conv2d_5/Conv2D/ReadVariableOp(^decoder/conv2d_6/BiasAdd/ReadVariableOp'^decoder/conv2d_6/Conv2D/ReadVariableOp'^decoder/dense_1/BiasAdd/ReadVariableOp&^decoder/dense_1/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp(^encoder/conv2d_2/BiasAdd/ReadVariableOp'^encoder/conv2d_2/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::2R
'decoder/conv2d_3/BiasAdd/ReadVariableOp'decoder/conv2d_3/BiasAdd/ReadVariableOp2P
&decoder/conv2d_3/Conv2D/ReadVariableOp&decoder/conv2d_3/Conv2D/ReadVariableOp2R
'decoder/conv2d_4/BiasAdd/ReadVariableOp'decoder/conv2d_4/BiasAdd/ReadVariableOp2P
&decoder/conv2d_4/Conv2D/ReadVariableOp&decoder/conv2d_4/Conv2D/ReadVariableOp2R
'decoder/conv2d_5/BiasAdd/ReadVariableOp'decoder/conv2d_5/BiasAdd/ReadVariableOp2P
&decoder/conv2d_5/Conv2D/ReadVariableOp&decoder/conv2d_5/Conv2D/ReadVariableOp2R
'decoder/conv2d_6/BiasAdd/ReadVariableOp'decoder/conv2d_6/BiasAdd/ReadVariableOp2P
&decoder/conv2d_6/Conv2D/ReadVariableOp&decoder/conv2d_6/Conv2D/ReadVariableOp2P
&decoder/dense_1/BiasAdd/ReadVariableOp&decoder/dense_1/BiasAdd/ReadVariableOp2N
%decoder/dense_1/MatMul/ReadVariableOp%decoder/dense_1/MatMul/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2R
'encoder/conv2d_2/BiasAdd/ReadVariableOp'encoder/conv2d_2/BiasAdd/ReadVariableOp2P
&encoder/conv2d_2/Conv2D/ReadVariableOp&encoder/conv2d_2/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

*__inference_conv2d_5_layer_call_fn_1594277

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *N
fIRG
E__inference_conv2d_5_layer_call_and_return_conditional_losses_15929012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1594120

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:???????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*8
_input_shapes'
%:???????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_1594212

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:????????? 2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:????????? 2

Identity"
identityIdentity:output:0*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
M
1__inference_up_sampling2d_2_layer_call_fn_1592782

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *>
config_proto.,

CPU

GPU2*0,1,2,3,4,5,6,7J 8? *U
fPRN
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_15927762
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1594288

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1592929

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd{
SigmoidSigmoidBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_1592433
input_1=
9autoencoder_encoder_conv2d_conv2d_readvariableop_resource>
:autoencoder_encoder_conv2d_biasadd_readvariableop_resource?
;autoencoder_encoder_conv2d_1_conv2d_readvariableop_resource@
<autoencoder_encoder_conv2d_1_biasadd_readvariableop_resource?
;autoencoder_encoder_conv2d_2_conv2d_readvariableop_resource@
<autoencoder_encoder_conv2d_2_biasadd_readvariableop_resource<
8autoencoder_encoder_dense_matmul_readvariableop_resource=
9autoencoder_encoder_dense_biasadd_readvariableop_resource>
:autoencoder_decoder_dense_1_matmul_readvariableop_resource?
;autoencoder_decoder_dense_1_biasadd_readvariableop_resource?
;autoencoder_decoder_conv2d_3_conv2d_readvariableop_resource@
<autoencoder_decoder_conv2d_3_biasadd_readvariableop_resource?
;autoencoder_decoder_conv2d_4_conv2d_readvariableop_resource@
<autoencoder_decoder_conv2d_4_biasadd_readvariableop_resource?
;autoencoder_decoder_conv2d_5_conv2d_readvariableop_resource@
<autoencoder_decoder_conv2d_5_biasadd_readvariableop_resource?
;autoencoder_decoder_conv2d_6_conv2d_readvariableop_resource@
<autoencoder_decoder_conv2d_6_biasadd_readvariableop_resource
identity??3autoencoder/decoder/conv2d_3/BiasAdd/ReadVariableOp?2autoencoder/decoder/conv2d_3/Conv2D/ReadVariableOp?3autoencoder/decoder/conv2d_4/BiasAdd/ReadVariableOp?2autoencoder/decoder/conv2d_4/Conv2D/ReadVariableOp?3autoencoder/decoder/conv2d_5/BiasAdd/ReadVariableOp?2autoencoder/decoder/conv2d_5/Conv2D/ReadVariableOp?3autoencoder/decoder/conv2d_6/BiasAdd/ReadVariableOp?2autoencoder/decoder/conv2d_6/Conv2D/ReadVariableOp?2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp?1autoencoder/decoder/dense_1/MatMul/ReadVariableOp?1autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp?0autoencoder/encoder/conv2d/Conv2D/ReadVariableOp?3autoencoder/encoder/conv2d_1/BiasAdd/ReadVariableOp?2autoencoder/encoder/conv2d_1/Conv2D/ReadVariableOp?3autoencoder/encoder/conv2d_2/BiasAdd/ReadVariableOp?2autoencoder/encoder/conv2d_2/Conv2D/ReadVariableOp?0autoencoder/encoder/dense/BiasAdd/ReadVariableOp?/autoencoder/encoder/dense/MatMul/ReadVariableOp?
0autoencoder/encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp9autoencoder_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0autoencoder/encoder/conv2d/Conv2D/ReadVariableOp?
!autoencoder/encoder/conv2d/Conv2DConv2Dinput_18autoencoder/encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!autoencoder/encoder/conv2d/Conv2D?
1autoencoder/encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp:autoencoder_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp?
"autoencoder/encoder/conv2d/BiasAddBiasAdd*autoencoder/encoder/conv2d/Conv2D:output:09autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2$
"autoencoder/encoder/conv2d/BiasAdd?
autoencoder/encoder/conv2d/ReluRelu+autoencoder/encoder/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2!
autoencoder/encoder/conv2d/Relu?
)autoencoder/encoder/max_pooling2d/MaxPoolMaxPool-autoencoder/encoder/conv2d/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2+
)autoencoder/encoder/max_pooling2d/MaxPool?
2autoencoder/encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;autoencoder_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2autoencoder/encoder/conv2d_1/Conv2D/ReadVariableOp?
#autoencoder/encoder/conv2d_1/Conv2DConv2D2autoencoder/encoder/max_pooling2d/MaxPool:output:0:autoencoder/encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#autoencoder/encoder/conv2d_1/Conv2D?
3autoencoder/encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3autoencoder/encoder/conv2d_1/BiasAdd/ReadVariableOp?
$autoencoder/encoder/conv2d_1/BiasAddBiasAdd,autoencoder/encoder/conv2d_1/Conv2D:output:0;autoencoder/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$autoencoder/encoder/conv2d_1/BiasAdd?
!autoencoder/encoder/conv2d_1/ReluRelu-autoencoder/encoder/conv2d_1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2#
!autoencoder/encoder/conv2d_1/Relu?
+autoencoder/encoder/max_pooling2d_1/MaxPoolMaxPool/autoencoder/encoder/conv2d_1/Relu:activations:0*/
_output_shapes
:?????????*
ksize
*
paddingSAME*
strides
2-
+autoencoder/encoder/max_pooling2d_1/MaxPool?
2autoencoder/encoder/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;autoencoder_encoder_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2autoencoder/encoder/conv2d_2/Conv2D/ReadVariableOp?
#autoencoder/encoder/conv2d_2/Conv2DConv2D4autoencoder/encoder/max_pooling2d_1/MaxPool:output:0:autoencoder/encoder/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#autoencoder/encoder/conv2d_2/Conv2D?
3autoencoder/encoder/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3autoencoder/encoder/conv2d_2/BiasAdd/ReadVariableOp?
$autoencoder/encoder/conv2d_2/BiasAddBiasAdd,autoencoder/encoder/conv2d_2/Conv2D:output:0;autoencoder/encoder/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2&
$autoencoder/encoder/conv2d_2/BiasAdd?
!autoencoder/encoder/conv2d_2/ReluRelu-autoencoder/encoder/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2#
!autoencoder/encoder/conv2d_2/Relu?
+autoencoder/encoder/max_pooling2d_2/MaxPoolMaxPool/autoencoder/encoder/conv2d_2/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingSAME*
strides
2-
+autoencoder/encoder/max_pooling2d_2/MaxPool?
!autoencoder/encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2#
!autoencoder/encoder/flatten/Const?
#autoencoder/encoder/flatten/ReshapeReshape4autoencoder/encoder/max_pooling2d_2/MaxPool:output:0*autoencoder/encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2%
#autoencoder/encoder/flatten/Reshape?
/autoencoder/encoder/dense/MatMul/ReadVariableOpReadVariableOp8autoencoder_encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype021
/autoencoder/encoder/dense/MatMul/ReadVariableOp?
 autoencoder/encoder/dense/MatMulMatMul,autoencoder/encoder/flatten/Reshape:output:07autoencoder/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 autoencoder/encoder/dense/MatMul?
0autoencoder/encoder/dense/BiasAdd/ReadVariableOpReadVariableOp9autoencoder_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0autoencoder/encoder/dense/BiasAdd/ReadVariableOp?
!autoencoder/encoder/dense/BiasAddBiasAdd*autoencoder/encoder/dense/MatMul:product:08autoencoder/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!autoencoder/encoder/dense/BiasAdd?
1autoencoder/decoder/dense_1/MatMul/ReadVariableOpReadVariableOp:autoencoder_decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype023
1autoencoder/decoder/dense_1/MatMul/ReadVariableOp?
"autoencoder/decoder/dense_1/MatMulMatMul*autoencoder/encoder/dense/BiasAdd:output:09autoencoder/decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2$
"autoencoder/decoder/dense_1/MatMul?
2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp;autoencoder_decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype024
2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp?
#autoencoder/decoder/dense_1/BiasAddBiasAdd,autoencoder/decoder/dense_1/MatMul:product:0:autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2%
#autoencoder/decoder/dense_1/BiasAdd?
!autoencoder/decoder/reshape/ShapeShape,autoencoder/decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2#
!autoencoder/decoder/reshape/Shape?
/autoencoder/decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/autoencoder/decoder/reshape/strided_slice/stack?
1autoencoder/decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1autoencoder/decoder/reshape/strided_slice/stack_1?
1autoencoder/decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1autoencoder/decoder/reshape/strided_slice/stack_2?
)autoencoder/decoder/reshape/strided_sliceStridedSlice*autoencoder/decoder/reshape/Shape:output:08autoencoder/decoder/reshape/strided_slice/stack:output:0:autoencoder/decoder/reshape/strided_slice/stack_1:output:0:autoencoder/decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)autoencoder/decoder/reshape/strided_slice?
+autoencoder/decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+autoencoder/decoder/reshape/Reshape/shape/1?
+autoencoder/decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+autoencoder/decoder/reshape/Reshape/shape/2?
+autoencoder/decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2-
+autoencoder/decoder/reshape/Reshape/shape/3?
)autoencoder/decoder/reshape/Reshape/shapePack2autoencoder/decoder/reshape/strided_slice:output:04autoencoder/decoder/reshape/Reshape/shape/1:output:04autoencoder/decoder/reshape/Reshape/shape/2:output:04autoencoder/decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2+
)autoencoder/decoder/reshape/Reshape/shape?
#autoencoder/decoder/reshape/ReshapeReshape,autoencoder/decoder/dense_1/BiasAdd:output:02autoencoder/decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2%
#autoencoder/decoder/reshape/Reshape?
2autoencoder/decoder/conv2d_3/Conv2D/ReadVariableOpReadVariableOp;autoencoder_decoder_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype024
2autoencoder/decoder/conv2d_3/Conv2D/ReadVariableOp?
#autoencoder/decoder/conv2d_3/Conv2DConv2D,autoencoder/decoder/reshape/Reshape:output:0:autoencoder/decoder/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#autoencoder/decoder/conv2d_3/Conv2D?
3autoencoder/decoder/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3autoencoder/decoder/conv2d_3/BiasAdd/ReadVariableOp?
$autoencoder/decoder/conv2d_3/BiasAddBiasAdd,autoencoder/decoder/conv2d_3/Conv2D:output:0;autoencoder/decoder/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2&
$autoencoder/decoder/conv2d_3/BiasAdd?
!autoencoder/decoder/conv2d_3/ReluRelu-autoencoder/decoder/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2#
!autoencoder/decoder/conv2d_3/Relu?
'autoencoder/decoder/up_sampling2d/ShapeShape/autoencoder/decoder/conv2d_3/Relu:activations:0*
T0*
_output_shapes
:2)
'autoencoder/decoder/up_sampling2d/Shape?
5autoencoder/decoder/up_sampling2d/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:27
5autoencoder/decoder/up_sampling2d/strided_slice/stack?
7autoencoder/decoder/up_sampling2d/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder/decoder/up_sampling2d/strided_slice/stack_1?
7autoencoder/decoder/up_sampling2d/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder/decoder/up_sampling2d/strided_slice/stack_2?
/autoencoder/decoder/up_sampling2d/strided_sliceStridedSlice0autoencoder/decoder/up_sampling2d/Shape:output:0>autoencoder/decoder/up_sampling2d/strided_slice/stack:output:0@autoencoder/decoder/up_sampling2d/strided_slice/stack_1:output:0@autoencoder/decoder/up_sampling2d/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:21
/autoencoder/decoder/up_sampling2d/strided_slice?
'autoencoder/decoder/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2)
'autoencoder/decoder/up_sampling2d/Const?
%autoencoder/decoder/up_sampling2d/mulMul8autoencoder/decoder/up_sampling2d/strided_slice:output:00autoencoder/decoder/up_sampling2d/Const:output:0*
T0*
_output_shapes
:2'
%autoencoder/decoder/up_sampling2d/mul?
>autoencoder/decoder/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor/autoencoder/decoder/conv2d_3/Relu:activations:0)autoencoder/decoder/up_sampling2d/mul:z:0*
T0*/
_output_shapes
:????????? *
half_pixel_centers(2@
>autoencoder/decoder/up_sampling2d/resize/ResizeNearestNeighbor?
2autoencoder/decoder/conv2d_4/Conv2D/ReadVariableOpReadVariableOp;autoencoder_decoder_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype024
2autoencoder/decoder/conv2d_4/Conv2D/ReadVariableOp?
#autoencoder/decoder/conv2d_4/Conv2DConv2DOautoencoder/decoder/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0:autoencoder/decoder/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#autoencoder/decoder/conv2d_4/Conv2D?
3autoencoder/decoder/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3autoencoder/decoder/conv2d_4/BiasAdd/ReadVariableOp?
$autoencoder/decoder/conv2d_4/BiasAddBiasAdd,autoencoder/decoder/conv2d_4/Conv2D:output:0;autoencoder/decoder/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2&
$autoencoder/decoder/conv2d_4/BiasAdd?
!autoencoder/decoder/conv2d_4/ReluRelu-autoencoder/decoder/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2#
!autoencoder/decoder/conv2d_4/Relu?
)autoencoder/decoder/up_sampling2d_1/ShapeShape/autoencoder/decoder/conv2d_4/Relu:activations:0*
T0*
_output_shapes
:2+
)autoencoder/decoder/up_sampling2d_1/Shape?
7autoencoder/decoder/up_sampling2d_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder/decoder/up_sampling2d_1/strided_slice/stack?
9autoencoder/decoder/up_sampling2d_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9autoencoder/decoder/up_sampling2d_1/strided_slice/stack_1?
9autoencoder/decoder/up_sampling2d_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9autoencoder/decoder/up_sampling2d_1/strided_slice/stack_2?
1autoencoder/decoder/up_sampling2d_1/strided_sliceStridedSlice2autoencoder/decoder/up_sampling2d_1/Shape:output:0@autoencoder/decoder/up_sampling2d_1/strided_slice/stack:output:0Bautoencoder/decoder/up_sampling2d_1/strided_slice/stack_1:output:0Bautoencoder/decoder/up_sampling2d_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1autoencoder/decoder/up_sampling2d_1/strided_slice?
)autoencoder/decoder/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)autoencoder/decoder/up_sampling2d_1/Const?
'autoencoder/decoder/up_sampling2d_1/mulMul:autoencoder/decoder/up_sampling2d_1/strided_slice:output:02autoencoder/decoder/up_sampling2d_1/Const:output:0*
T0*
_output_shapes
:2)
'autoencoder/decoder/up_sampling2d_1/mul?
@autoencoder/decoder/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor/autoencoder/decoder/conv2d_4/Relu:activations:0+autoencoder/decoder/up_sampling2d_1/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2B
@autoencoder/decoder/up_sampling2d_1/resize/ResizeNearestNeighbor?
2autoencoder/decoder/conv2d_5/Conv2D/ReadVariableOpReadVariableOp;autoencoder_decoder_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2autoencoder/decoder/conv2d_5/Conv2D/ReadVariableOp?
#autoencoder/decoder/conv2d_5/Conv2DConv2DQautoencoder/decoder/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0:autoencoder/decoder/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#autoencoder/decoder/conv2d_5/Conv2D?
3autoencoder/decoder/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3autoencoder/decoder/conv2d_5/BiasAdd/ReadVariableOp?
$autoencoder/decoder/conv2d_5/BiasAddBiasAdd,autoencoder/decoder/conv2d_5/Conv2D:output:0;autoencoder/decoder/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$autoencoder/decoder/conv2d_5/BiasAdd?
!autoencoder/decoder/conv2d_5/ReluRelu-autoencoder/decoder/conv2d_5/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2#
!autoencoder/decoder/conv2d_5/Relu?
)autoencoder/decoder/up_sampling2d_2/ShapeShape/autoencoder/decoder/conv2d_5/Relu:activations:0*
T0*
_output_shapes
:2+
)autoencoder/decoder/up_sampling2d_2/Shape?
7autoencoder/decoder/up_sampling2d_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:29
7autoencoder/decoder/up_sampling2d_2/strided_slice/stack?
9autoencoder/decoder/up_sampling2d_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9autoencoder/decoder/up_sampling2d_2/strided_slice/stack_1?
9autoencoder/decoder/up_sampling2d_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9autoencoder/decoder/up_sampling2d_2/strided_slice/stack_2?
1autoencoder/decoder/up_sampling2d_2/strided_sliceStridedSlice2autoencoder/decoder/up_sampling2d_2/Shape:output:0@autoencoder/decoder/up_sampling2d_2/strided_slice/stack:output:0Bautoencoder/decoder/up_sampling2d_2/strided_slice/stack_1:output:0Bautoencoder/decoder/up_sampling2d_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:23
1autoencoder/decoder/up_sampling2d_2/strided_slice?
)autoencoder/decoder/up_sampling2d_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2+
)autoencoder/decoder/up_sampling2d_2/Const?
'autoencoder/decoder/up_sampling2d_2/mulMul:autoencoder/decoder/up_sampling2d_2/strided_slice:output:02autoencoder/decoder/up_sampling2d_2/Const:output:0*
T0*
_output_shapes
:2)
'autoencoder/decoder/up_sampling2d_2/mul?
@autoencoder/decoder/up_sampling2d_2/resize/ResizeNearestNeighborResizeNearestNeighbor/autoencoder/decoder/conv2d_5/Relu:activations:0+autoencoder/decoder/up_sampling2d_2/mul:z:0*
T0*1
_output_shapes
:???????????*
half_pixel_centers(2B
@autoencoder/decoder/up_sampling2d_2/resize/ResizeNearestNeighbor?
2autoencoder/decoder/conv2d_6/Conv2D/ReadVariableOpReadVariableOp;autoencoder_decoder_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype024
2autoencoder/decoder/conv2d_6/Conv2D/ReadVariableOp?
#autoencoder/decoder/conv2d_6/Conv2DConv2DQautoencoder/decoder/up_sampling2d_2/resize/ResizeNearestNeighbor:resized_images:0:autoencoder/decoder/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2%
#autoencoder/decoder/conv2d_6/Conv2D?
3autoencoder/decoder/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp<autoencoder_decoder_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3autoencoder/decoder/conv2d_6/BiasAdd/ReadVariableOp?
$autoencoder/decoder/conv2d_6/BiasAddBiasAdd,autoencoder/decoder/conv2d_6/Conv2D:output:0;autoencoder/decoder/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2&
$autoencoder/decoder/conv2d_6/BiasAdd?
$autoencoder/decoder/conv2d_6/SigmoidSigmoid-autoencoder/decoder/conv2d_6/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2&
$autoencoder/decoder/conv2d_6/Sigmoid?
IdentityIdentity(autoencoder/decoder/conv2d_6/Sigmoid:y:04^autoencoder/decoder/conv2d_3/BiasAdd/ReadVariableOp3^autoencoder/decoder/conv2d_3/Conv2D/ReadVariableOp4^autoencoder/decoder/conv2d_4/BiasAdd/ReadVariableOp3^autoencoder/decoder/conv2d_4/Conv2D/ReadVariableOp4^autoencoder/decoder/conv2d_5/BiasAdd/ReadVariableOp3^autoencoder/decoder/conv2d_5/Conv2D/ReadVariableOp4^autoencoder/decoder/conv2d_6/BiasAdd/ReadVariableOp3^autoencoder/decoder/conv2d_6/Conv2D/ReadVariableOp3^autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp2^autoencoder/decoder/dense_1/MatMul/ReadVariableOp2^autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp1^autoencoder/encoder/conv2d/Conv2D/ReadVariableOp4^autoencoder/encoder/conv2d_1/BiasAdd/ReadVariableOp3^autoencoder/encoder/conv2d_1/Conv2D/ReadVariableOp4^autoencoder/encoder/conv2d_2/BiasAdd/ReadVariableOp3^autoencoder/encoder/conv2d_2/Conv2D/ReadVariableOp1^autoencoder/encoder/dense/BiasAdd/ReadVariableOp0^autoencoder/encoder/dense/MatMul/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*x
_input_shapesg
e:???????????::::::::::::::::::2j
3autoencoder/decoder/conv2d_3/BiasAdd/ReadVariableOp3autoencoder/decoder/conv2d_3/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/conv2d_3/Conv2D/ReadVariableOp2autoencoder/decoder/conv2d_3/Conv2D/ReadVariableOp2j
3autoencoder/decoder/conv2d_4/BiasAdd/ReadVariableOp3autoencoder/decoder/conv2d_4/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/conv2d_4/Conv2D/ReadVariableOp2autoencoder/decoder/conv2d_4/Conv2D/ReadVariableOp2j
3autoencoder/decoder/conv2d_5/BiasAdd/ReadVariableOp3autoencoder/decoder/conv2d_5/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/conv2d_5/Conv2D/ReadVariableOp2autoencoder/decoder/conv2d_5/Conv2D/ReadVariableOp2j
3autoencoder/decoder/conv2d_6/BiasAdd/ReadVariableOp3autoencoder/decoder/conv2d_6/BiasAdd/ReadVariableOp2h
2autoencoder/decoder/conv2d_6/Conv2D/ReadVariableOp2autoencoder/decoder/conv2d_6/Conv2D/ReadVariableOp2h
2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp2autoencoder/decoder/dense_1/BiasAdd/ReadVariableOp2f
1autoencoder/decoder/dense_1/MatMul/ReadVariableOp1autoencoder/decoder/dense_1/MatMul/ReadVariableOp2f
1autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp1autoencoder/encoder/conv2d/BiasAdd/ReadVariableOp2d
0autoencoder/encoder/conv2d/Conv2D/ReadVariableOp0autoencoder/encoder/conv2d/Conv2D/ReadVariableOp2j
3autoencoder/encoder/conv2d_1/BiasAdd/ReadVariableOp3autoencoder/encoder/conv2d_1/BiasAdd/ReadVariableOp2h
2autoencoder/encoder/conv2d_1/Conv2D/ReadVariableOp2autoencoder/encoder/conv2d_1/Conv2D/ReadVariableOp2j
3autoencoder/encoder/conv2d_2/BiasAdd/ReadVariableOp3autoencoder/encoder/conv2d_2/BiasAdd/ReadVariableOp2h
2autoencoder/encoder/conv2d_2/Conv2D/ReadVariableOp2autoencoder/encoder/conv2d_2/Conv2D/ReadVariableOp2d
0autoencoder/encoder/dense/BiasAdd/ReadVariableOp0autoencoder/encoder/dense/BiasAdd/ReadVariableOp2b
/autoencoder/encoder/dense/MatMul/ReadVariableOp/autoencoder/encoder/dense/MatMul/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????E
decoder:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
٢
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	optimizer

signatures
#_self_saveable_object_factories
trainable_variables
	variables
	regularization_losses

	keras_api
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"??
_tf_keras_network??{"class_name": "Functional", "name": "autoencoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "name": "encoder", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 32]}}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_6", 0, 0]]}, "name": "decoder", "inbound_nodes": [[["encoder", 1, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["decoder", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 1024, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "autoencoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "name": "encoder", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 32]}}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_6", 0, 0]]}, "name": "decoder", "inbound_nodes": [[["encoder", 1, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["decoder", 1, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
#_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?K
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
layer_with_weights-3
layer-8
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?H
_tf_keras_network?G{"class_name": "Functional", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 1024, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1024, 1024, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["max_pooling2d", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "name": "max_pooling2d_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["max_pooling2d_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "max_pooling2d_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatten", "inbound_nodes": [[["max_pooling2d_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["flatten", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
?T
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
 layer_with_weights-3
 layer-7
!layer-8
"layer_with_weights-4
"layer-9
##_self_saveable_object_factories
$trainable_variables
%	variables
&regularization_losses
'	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?Q
_tf_keras_network?Q{"class_name": "Functional", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 32]}}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_6", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "decoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 32]}}, "name": "reshape", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["up_sampling2d", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_1", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_5", "inbound_nodes": [[["up_sampling2d_1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_2", "inbound_nodes": [[["conv2d_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_6", "inbound_nodes": [[["up_sampling2d_2", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["conv2d_6", 0, 0]]}}}
?
(iter

)beta_1

*beta_2
	+decay
,learning_rate-m?.m?/m?0m?1m?2m?3m?4m?5m?6m?7m?8m?9m?:m?;m?<m?=m?>m?-v?.v?/v?0v?1v?2v?3v?4v?5v?6v?7v?8v?9v?:v?;v?<v?=v?>v?"
	optimizer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17"
trackable_list_wrapper
?
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
@layer_regularization_losses
Alayer_metrics
trainable_variables
	variables

Blayers
Cnon_trainable_variables
	regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
?


-kernel
.bias
#D_self_saveable_object_factories
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 1024, 1]}}
?
#I_self_saveable_object_factories
Jregularization_losses
Ktrainable_variables
L	variables
M	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


/kernel
0bias
#N_self_saveable_object_factories
Oregularization_losses
Ptrainable_variables
Q	variables
R	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 8]}}
?
#S_self_saveable_object_factories
Tregularization_losses
Utrainable_variables
V	variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [8, 8]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


1kernel
2bias
#X_self_saveable_object_factories
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 16]}}
?
#]_self_saveable_object_factories
^regularization_losses
_trainable_variables
`	variables
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "MaxPooling2D", "name": "max_pooling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?
#b_self_saveable_object_factories
cregularization_losses
dtrainable_variables
e	variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

3kernel
4bias
#g_self_saveable_object_factories
hregularization_losses
itrainable_variables
j	variables
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512]}}
 "
trackable_dict_wrapper
X
-0
.1
/2
03
14
25
36
47"
trackable_list_wrapper
X
-0
.1
/2
03
14
25
36
47"
trackable_list_wrapper
 "
trackable_list_wrapper
?
lmetrics
mlayer_regularization_losses
nlayer_metrics
trainable_variables
	variables

olayers
pnon_trainable_variables
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
#q_self_saveable_object_factories"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
?

5kernel
6bias
#r_self_saveable_object_factories
sregularization_losses
ttrainable_variables
u	variables
v	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 512, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
?
#w_self_saveable_object_factories
xregularization_losses
ytrainable_variables
z	variables
{	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4, 4, 32]}}}
?


7kernel
8bias
#|_self_saveable_object_factories
}regularization_losses
~trainable_variables
	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4, 4, 32]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


9kernel
:bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [8, 8]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


;kernel
<bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 16]}}
?
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "UpSampling2D", "name": "up_sampling2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_2", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [8, 8]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
?


=kernel
>bias
$?_self_saveable_object_factories
?regularization_losses
?trainable_variables
?	variables
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [16, 16]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1024, 1024, 8]}}
 "
trackable_dict_wrapper
f
50
61
72
83
94
:5
;6
<7
=8
>9"
trackable_list_wrapper
f
50
61
72
83
94
:5
;6
<7
=8
>9"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?metrics
 ?layer_regularization_losses
?layer_metrics
$trainable_variables
%	variables
?layers
?non_trainable_variables
&regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 :2kernel
:2bias
 :2kernel
:2bias
 : 2kernel
: 2bias
:	?2kernel
:2bias
:	?2kernel
:?2bias
 :  2kernel
: 2bias
 : 2kernel
:2bias
 :2kernel
:2bias
 :2kernel
:2bias
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
?
Eregularization_losses
?layer_metrics
?layers
Ftrainable_variables
G	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Jregularization_losses
?layer_metrics
?layers
Ktrainable_variables
L	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
Oregularization_losses
?layer_metrics
?layers
Ptrainable_variables
Q	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Tregularization_losses
?layer_metrics
?layers
Utrainable_variables
V	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
?
Yregularization_losses
?layer_metrics
?layers
Ztrainable_variables
[	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^regularization_losses
?layer_metrics
?layers
_trainable_variables
`	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
cregularization_losses
?layer_metrics
?layers
dtrainable_variables
e	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
?
hregularization_losses
?layer_metrics
?layers
itrainable_variables
j	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
sregularization_losses
?layer_metrics
?layers
ttrainable_variables
u	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
xregularization_losses
?layer_metrics
?layers
ytrainable_variables
z	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
?
}regularization_losses
?layer_metrics
?layers
~trainable_variables
	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
?
?regularization_losses
?layer_metrics
?layers
?trainable_variables
?	variables
?non_trainable_variables
?metrics
 ?layer_regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
f
0
1
2
3
4
5
6
 7
!8
"9"
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
%:#2Adam/kernel/m
:2Adam/bias/m
%:#2Adam/kernel/m
:2Adam/bias/m
%:# 2Adam/kernel/m
: 2Adam/bias/m
:	?2Adam/kernel/m
:2Adam/bias/m
:	?2Adam/kernel/m
:?2Adam/bias/m
%:#  2Adam/kernel/m
: 2Adam/bias/m
%:# 2Adam/kernel/m
:2Adam/bias/m
%:#2Adam/kernel/m
:2Adam/bias/m
%:#2Adam/kernel/m
:2Adam/bias/m
%:#2Adam/kernel/v
:2Adam/bias/v
%:#2Adam/kernel/v
:2Adam/bias/v
%:# 2Adam/kernel/v
: 2Adam/bias/v
:	?2Adam/kernel/v
:2Adam/bias/v
:	?2Adam/kernel/v
:?2Adam/bias/v
%:#  2Adam/kernel/v
: 2Adam/bias/v
%:# 2Adam/kernel/v
:2Adam/bias/v
%:#2Adam/kernel/v
:2Adam/bias/v
%:#2Adam/kernel/v
:2Adam/bias/v
?2?
-__inference_autoencoder_layer_call_fn_1593440
-__inference_autoencoder_layer_call_fn_1593740
-__inference_autoencoder_layer_call_fn_1593781
-__inference_autoencoder_layer_call_fn_1593357?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_1592433?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *0?-
+?(
input_1???????????
?2?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593595
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593699
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593231
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593273?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_encoder_layer_call_fn_1593874
)__inference_encoder_layer_call_fn_1593895
)__inference_encoder_layer_call_fn_1592676
)__inference_encoder_layer_call_fn_1592725?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_encoder_layer_call_and_return_conditional_losses_1592626
D__inference_encoder_layer_call_and_return_conditional_losses_1593853
D__inference_encoder_layer_call_and_return_conditional_losses_1593817
D__inference_encoder_layer_call_and_return_conditional_losses_1592598?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_decoder_layer_call_fn_1594064
)__inference_decoder_layer_call_fn_1593096
)__inference_decoder_layer_call_fn_1594089
)__inference_decoder_layer_call_fn_1593038?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_decoder_layer_call_and_return_conditional_losses_1594039
D__inference_decoder_layer_call_and_return_conditional_losses_1593967
D__inference_decoder_layer_call_and_return_conditional_losses_1592946
D__inference_decoder_layer_call_and_return_conditional_losses_1592979?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
%__inference_signature_wrapper_1593491input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_conv2d_layer_call_fn_1594109?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_conv2d_layer_call_and_return_conditional_losses_1594100?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_max_pooling2d_layer_call_fn_1592445?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1592439?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_1_layer_call_fn_1594129?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1594120?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling2d_1_layer_call_fn_1592457?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1592451?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_2_layer_call_fn_1594149?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1594140?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_max_pooling2d_2_layer_call_fn_1592469?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1592463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
)__inference_flatten_layer_call_fn_1594160?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_1594155?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_1594179?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_1594170?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_1594198?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_1594189?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_layer_call_fn_1594217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_layer_call_and_return_conditional_losses_1594212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2d_3_layer_call_fn_1594237?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1594228?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_up_sampling2d_layer_call_fn_1592744?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1592738?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_4_layer_call_fn_1594257?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1594248?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_up_sampling2d_1_layer_call_fn_1592763?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1592757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_5_layer_call_fn_1594277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1594268?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_up_sampling2d_2_layer_call_fn_1592782?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1592776?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
*__inference_conv2d_6_layer_call_fn_1594297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1594288?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_1592433?-./0123456789:;<=>:?7
0?-
+?(
input_1???????????
? ";?8
6
decoder+?(
decoder????????????
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593231?-./0123456789:;<=>B??
8?5
+?(
input_1???????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593273?-./0123456789:;<=>B??
8?5
+?(
input_1???????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593595?-./0123456789:;<=>A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
H__inference_autoencoder_layer_call_and_return_conditional_losses_1593699?-./0123456789:;<=>A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
-__inference_autoencoder_layer_call_fn_1593357?-./0123456789:;<=>B??
8?5
+?(
input_1???????????
p

 
? "2?/+????????????????????????????
-__inference_autoencoder_layer_call_fn_1593440?-./0123456789:;<=>B??
8?5
+?(
input_1???????????
p 

 
? "2?/+????????????????????????????
-__inference_autoencoder_layer_call_fn_1593740?-./0123456789:;<=>A?>
7?4
*?'
inputs???????????
p

 
? "2?/+????????????????????????????
-__inference_autoencoder_layer_call_fn_1593781?-./0123456789:;<=>A?>
7?4
*?'
inputs???????????
p 

 
? "2?/+????????????????????????????
E__inference_conv2d_1_layer_call_and_return_conditional_losses_1594120p/09?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
*__inference_conv2d_1_layer_call_fn_1594129c/09?6
/?,
*?'
inputs???????????
? ""?????????????
E__inference_conv2d_2_layer_call_and_return_conditional_losses_1594140l127?4
-?*
(?%
inputs?????????
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_2_layer_call_fn_1594149_127?4
-?*
(?%
inputs?????????
? " ?????????? ?
E__inference_conv2d_3_layer_call_and_return_conditional_losses_1594228l787?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
*__inference_conv2d_3_layer_call_fn_1594237_787?4
-?*
(?%
inputs????????? 
? " ?????????? ?
E__inference_conv2d_4_layer_call_and_return_conditional_losses_1594248?9:I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
*__inference_conv2d_4_layer_call_fn_1594257?9:I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
E__inference_conv2d_5_layer_call_and_return_conditional_losses_1594268?;<I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
*__inference_conv2d_5_layer_call_fn_1594277?;<I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
E__inference_conv2d_6_layer_call_and_return_conditional_losses_1594288?=>I?F
??<
:?7
inputs+???????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
*__inference_conv2d_6_layer_call_fn_1594297?=>I?F
??<
:?7
inputs+???????????????????????????
? "2?/+????????????????????????????
C__inference_conv2d_layer_call_and_return_conditional_losses_1594100p-.9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
(__inference_conv2d_layer_call_fn_1594109c-.9?6
/?,
*?'
inputs???????????
? ""?????????????
D__inference_decoder_layer_call_and_return_conditional_losses_1592946?
56789:;<=>8?5
.?+
!?
input_2?????????
p

 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_1592979?
56789:;<=>8?5
.?+
!?
input_2?????????
p 

 
? "??<
5?2
0+???????????????????????????
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_1593967v
56789:;<=>7?4
-?*
 ?
inputs?????????
p

 
? "/?,
%?"
0???????????
? ?
D__inference_decoder_layer_call_and_return_conditional_losses_1594039v
56789:;<=>7?4
-?*
 ?
inputs?????????
p 

 
? "/?,
%?"
0???????????
? ?
)__inference_decoder_layer_call_fn_1593038z
56789:;<=>8?5
.?+
!?
input_2?????????
p

 
? "2?/+????????????????????????????
)__inference_decoder_layer_call_fn_1593096z
56789:;<=>8?5
.?+
!?
input_2?????????
p 

 
? "2?/+????????????????????????????
)__inference_decoder_layer_call_fn_1594064y
56789:;<=>7?4
-?*
 ?
inputs?????????
p

 
? "2?/+????????????????????????????
)__inference_decoder_layer_call_fn_1594089y
56789:;<=>7?4
-?*
 ?
inputs?????????
p 

 
? "2?/+????????????????????????????
D__inference_dense_1_layer_call_and_return_conditional_losses_1594189]56/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? }
)__inference_dense_1_layer_call_fn_1594198P56/?,
%?"
 ?
inputs?????????
? "????????????
B__inference_dense_layer_call_and_return_conditional_losses_1594170]340?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? {
'__inference_dense_layer_call_fn_1594179P340?-
&?#
!?
inputs??????????
? "???????????
D__inference_encoder_layer_call_and_return_conditional_losses_1592598u-./01234B??
8?5
+?(
input_1???????????
p

 
? "%?"
?
0?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_1592626u-./01234B??
8?5
+?(
input_1???????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_1593817t-./01234A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
D__inference_encoder_layer_call_and_return_conditional_losses_1593853t-./01234A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
)__inference_encoder_layer_call_fn_1592676h-./01234B??
8?5
+?(
input_1???????????
p

 
? "???????????
)__inference_encoder_layer_call_fn_1592725h-./01234B??
8?5
+?(
input_1???????????
p 

 
? "???????????
)__inference_encoder_layer_call_fn_1593874g-./01234A?>
7?4
*?'
inputs???????????
p

 
? "???????????
)__inference_encoder_layer_call_fn_1593895g-./01234A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
D__inference_flatten_layer_call_and_return_conditional_losses_1594155a7?4
-?*
(?%
inputs????????? 
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_1594160T7?4
-?*
(?%
inputs????????? 
? "????????????
L__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_1592451?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_1_layer_call_fn_1592457?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_1592463?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_max_pooling2d_2_layer_call_fn_1592469?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_max_pooling2d_layer_call_and_return_conditional_losses_1592439?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_max_pooling2d_layer_call_fn_1592445?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
D__inference_reshape_layer_call_and_return_conditional_losses_1594212a0?-
&?#
!?
inputs??????????
? "-?*
#? 
0????????? 
? ?
)__inference_reshape_layer_call_fn_1594217T0?-
&?#
!?
inputs??????????
? " ?????????? ?
%__inference_signature_wrapper_1593491?-./0123456789:;<=>E?B
? 
;?8
6
input_1+?(
input_1???????????";?8
6
decoder+?(
decoder????????????
L__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_1592757?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_up_sampling2d_1_layer_call_fn_1592763?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
L__inference_up_sampling2d_2_layer_call_and_return_conditional_losses_1592776?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
1__inference_up_sampling2d_2_layer_call_fn_1592782?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_up_sampling2d_layer_call_and_return_conditional_losses_1592738?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
/__inference_up_sampling2d_layer_call_fn_1592744?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????