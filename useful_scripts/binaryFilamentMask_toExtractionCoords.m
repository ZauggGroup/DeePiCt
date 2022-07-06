%%
% Takes binary filament mask, shrinks down to reduce contact points, traces skeleton
% samples points, traces along to order points, then draws splines and
% samples along to take points
% Skeleton3d needs to be added to the path(Skeleton3D.m installed, https://www.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d)
% Dynamo 1.1.520 needs to be running
%
% Fergus Tollervey - June 2022
%


%%
clear
%Shrinks to remove contacts between microtubules in generated volume
tomo=dread('post_processed_prediction.mrc');
shrinkFactor=ones(6,6,6); %value decided arbritatrararily
shrunk=imerode(tomo,shrinkFactor);
shrunk=logical(shrunk);

%remove noisy remaining bits
cleaned=bwareaopen(shrunk,400);
label=bwlabeln(cleaned,26);

%skeletonizes (needs Skeleton3D.m installed, https://www.mathworks.com/matlabcentral/fileexchange/43400-skeleton3d)
skel=Skeleton3D(cleaned);

%sort pixels into per-filament groups
skelLabel=bwlabeln(skel,26);
%max(skelLabel,[],'all')
dwrite(skelLabel,'skeletonLabels.mrc');


allFilamentsTable=[];
filamentNumber=max(skelLabel,[],'all');
for i=1:filamentNumber
        [x,y,z]=ind2sub(size(tomo),find(skelLabel==i)); %makes volume with just one filament
        individualFilament=zeros(size(tomo));
        for ii=1:size(x,1)
            X=x(ii,:);
            Y=y(ii,:);
            Z=z(ii,:);
            individualFilament(X,Y,Z)=1;
        end
        endpoints=bwmorph3(individualFilament,'endpoints'); %finds endpoint and moves on from there
        %find endpoint of individual filament
        [a,b,c]=ind2sub(size(tomo),find(endpoints==1));
        %make sure there are only 2 endpoints, remove all the others.
        %Prevents errant branching
        if size(a,1)>=3
            %scan down and remove branches
            removeBranchStart=[a(2:end-1), b(2:end-1), c(2:end-1)];
            for rm=1:size(removeBranchStart,1)
                rmX=removeBranchStart(rm,1);
                rmY=removeBranchStart(rm,2);
                rmZ=removeBranchStart(rm,3);
                individualFilament(rmX,rmY,rmZ)=0;
                %make subbox around coordinate point
                point=removeBranchStart;
                for branchLength=1:50 %arbritary, just because I hate using while
                    %remove current value from individualFilament
                    individualFilament(point(1),point(2),point(3))=0;
                    subbox=individualFilament(point(1)-1:point(1)+1 , point(2)-1:point(2)+1 , point(3)-1:point(3)+1);
                    %remove center
                    subbox(2,2,2)=0;

                    %find non-zero location
                    [branchA,branchB,branchC]=ind2sub(size(subbox),find(subbox==1));
                    if sum(subbox,'all')~=1
                        break
                    end
                    %convert non-zero location into an xyz shift
                    if branchA==2
                        branchA=0;
                    elseif branchA==1
                        branchA=-1;
                    else branchA=1;
                    end
                    if branchB==2
                        branchB=0;
                    elseif branchB==1
                        branchB=-1;
                    else branchB=1;
                    end
                    if branchC==2
                        branchC=0;
                    elseif branchC==1
                        branchC=-1;
                    else branchC=1;
                    end

                %shift to next coordinate
                point=point+[branchA branchB branchC];
                end

            end

        end

        coordinateList=[a(1,1),b(1,1),c(1,1)];
        for iii=1:size(x,1)
            point=coordinateList(iii,:);
            %make subbox around coordinate point
            subbox=individualFilament(point(1)-1:point(1)+1 , point(2)-1:point(2)+1 , point(3)-1:point(3)+1);
            %remove center
            subbox(2,2,2)=0;

            %find non-zero location
            [a,b,c]=ind2sub(size(subbox),find(subbox==1));
            if sum(subbox,'all')~=1
                break
            end
            %convert non-zero location into an xyz shift
            if a==2
                a=0;
            elseif a==1
                a=-1;
            else a=1;
            end
            if b==2
                b=0;
            elseif b==1
                b=-1;
            else b=1;
            end
            if c==2
                c=0;
            elseif c==1
                c=-1;
            else c=1;
            end
            %shift to next coordinate
            nextCoord=coordinateList(iii,:)+[a b c];
            %add current coordinate to table
            coordinateList=cat(1,coordinateList,nextCoord);
            %remove current value from individual filament
            individualFilament(point(1),point(2),point(3))=0;

        end
        %add filament ID to table
        coordinateList(:,4)=i;
        %add coordinates to full table
        allFilamentsTable=cat(1,allFilamentsTable,coordinateList);

end



dwrite(allFilamentsTable,'allFilamentsTable.em')

% reduce table size to smooth out splines
allFilamentsTableRed=allFilamentsTable(10:10:end,:);


%% below taken from my modified versions of Xiaojie's amira2star script

allFilamentsTableRed=allFilamentsTableRed';
motl_or=zeros(20,size(allFilamentsTableRed,2));

motl_or(8,:) = allFilamentsTableRed(1,:);
motl_or(9,:) = allFilamentsTableRed(2,:);
motl_or(10,:)= allFilamentsTableRed(3,:);
motl_or(4,:) = 1:size(allFilamentsTableRed,2);                % index of particle
motl_or(5,:) = 6;                                  % ntomo, arbritary and not kept anyway
motl_or(7,:) = allFilamentsTableRed(4,:);                     % index of filament

%% MOTL Processing

tt=unique(motl_or(7,:));
particleradius=0.1;
pixelstep=6;        % sampling on each turn

outputmotlname=['processedMotl.em'];
motl_acc=[];

for shtube=1:size(tt,2)
% for shtube=1
    tuben=tt(1,shtube);


    %generating spline fit of filament


    %%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%

    motl2=motl_or(:,motl_or(7,:)==tuben);



    m=size(motl2,2);
    F=spline((1:m),motl2(8:10,:));
    totalsteps=0;
    clear motlnew;
    for n=2:m
        xc=motl2(8,n)-motl2(8,(n-1));
        yc=motl2(9,n)-motl2(9,(n-1));
        zc=motl2(10,n)-motl2(10,(n-1));
        dist=(xc^2)+(yc^2)+(zc^2);
        dist=dist^0.5;
        stepnumber=round(dist/1);
        step=1/stepnumber;
        t=[n-1:step:n];
        Ft(:,totalsteps+1:totalsteps+size(t,2))=ppval(F,t);
        totalsteps=totalsteps+size(t,2);
    end
    c=1;
    while c < size(Ft,2)
        xc=Ft(1,c+1)-Ft(1,c);
        yc=Ft(2,c+1)-Ft(2,c);
        zc=Ft(3,c+1)-Ft(3,c);
        dist=(xc^2)+(yc^2)+(zc^2);
        dist=dist^0.5;
        if dist<pixelstep
            Ft(:,c+1)=[];
        else c=c+1;
        end
    end



    %generating surface positions

    numberofangles=1; %1 is for full MT cross section; if picking wall round(2*pi/angle);
    angle=2*pi/numberofangles;
    anglelist=[angle:angle:2*pi];

    for n=2:size(Ft,2)
        [THETA,PHI,R]=cart2sph(Ft(1,n)-Ft(1,n-1),Ft(2,n)-Ft(2,n-1),Ft(3,n)-Ft(3,n-1));
        ORIG=[Ft(1,n);Ft(2,n);Ft(3,n)];
        for angle=1:length(anglelist)
            VEC=[particleradius;0;0];
            %VEC=Rz(VEC,anglelist(angle)); %for fixed angle
            VEC=Rz(VEC,(n-2)*(360/13));  %  in-plane rotation of 27.76 deg per particle
            VEC=Ry(VEC,-PHI-(pi/2));
            VEC=Rz(VEC,THETA);
            VEC=VEC+ORIG;
            motlv=zeros(20,1);
            motlv(8,1)=VEC(1,1);
            motlv(9,1)=VEC(2,1);
            motlv(10,1)=VEC(3,1);
            motlv=normalvec(motlv,ORIG);



            vec=[Ft(1,n)-Ft(1,n-1) Ft(2,n)-Ft(2,n-1) Ft(3,n)-Ft(3,n-1)];
            vec=tom_pointrotate(vec,-motlv(18,1),0,-motlv(19,1));
            phi=360/(2*pi())*atan2(vec(2),vec(1));
            motlv(17,1)=phi;

            if exist('motlnew')==0
                 motlnew=motlv;
            else
                motlnew=[motlnew motlv];
            end
        end
    end

    motlnew(5,:)=motl_or(5,1);
    motlnew(7,:)=tuben;
    motlnew(20,:)=1;

    motl_acc=cat(2,motl_acc,motlnew);
    tt(2,shtube)= size(motlnew,2);

end

for i=1:size(motl_acc,2)
    motl_acc(4,i)=i;
end
%% Flip Particles to be orthogonal to the xy plane
% Dynamo 1.1.520 needs to be running
model=dmimport('motl',motl_acc);% convert motl to model
tbl=model.grepTable();%convert model to table
tbl=dynamo_table_rigid(tbl,[90,90,270]);%rigid transformation from yz orthogonal plane to xy orthogonal

%% Dynamo cropping and averaging steps
dwrite(tbl,'initialTable.tbl')
o = dtcrop(dread('/g/scb/mahamid/sgoetz/Data/Titan/Processing/HeLa/imod_rotated/t2_2b_rot.rec'),tbl,'particlesData_2bin',72,'allow_padding','true');
%o = dtcrop(dread('t2_forsegmentation.mrc'),tbl,'particlesData_4bin',36,'allow_padding','true');
oa=daverage('particlesData_2bin','t',tbl);
dwrite(oa.average,'initialAverage_2bin.mrc')
