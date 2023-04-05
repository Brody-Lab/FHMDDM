function C = colors()
% COLORS Default colors associated with task conditions or model settings
%
% C = COLORS() returns a structure whose name of each field is the name of a task condition or model
% setting, and the value of that field is a three-element array specifying the intensity of the red,
% green, and blue component of the color.
C = struct;
C.unconditioned = [0,0,0];
C.leftchoice = [0.229999504, 0.298998934, 0.754000139];
C.rightchoice = [0.706000136, 0.015991824, 0.150000072];
C.leftevidence = C.leftchoice;
C.rightevidence = C.rightchoice;
C.leftchoice_weak_leftevidence = [0.667602712               0.779706789               0.993625576];
C.leftchoice_strong_leftevidence = C.leftchoice;
C.rightchoice_weak_rightevidence = [0.968998399               0.721381489               0.612361865];
C.rightchoice_strong_rightevidence = C.rightchoice;
