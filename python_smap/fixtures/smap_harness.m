function out = smap_harness( f )
	S = load(f); X = S.X;
	X = fillmissing(X, 'linear', 'EndValues','nearest');
	opts = struct('bin',5,'E',3,'h',1,'theiler',8,'theta',[0 0.25 0.5 1 2 4 8 16 32],'ridge',1e-4);
	pairs = [1 2; 2 1; 3 2];
	Xb = local_bin(X, opts.bin);
	[rho, IS, tBin, AR] = local_smapAll(Xb, pairs, opts, true, true);
	[lam, tau] = mmv_smapLambda(AR);
	out = struct('Xb',Xb,'rho',rho,'IS',IS,'tBin',tBin,'AR',double(AR),'lam',lam,'tau',tau);
	save(strrep(f,'.mat','_ml.mat'), '-struct', 'out');
end
function Xb = local_bin( X, b )
	n = floor(size(X,1)/b)*b;
	Xb = squeeze(mean(reshape(X(1:n,:), b, n/b, size(X,2)), 1));
	Xb = zscore(Xb);
end

function [rho, IS, t, AR] = local_smapAll( Xb, pairs, opts, wantIS, wantAR )
	if nargin < 5, wantAR = false; end
	n = size(Xb,1); E = opts.E; h = opts.h; th = opts.theta; nTh = numel(th);
	t = (E:n-h).'; nT = numel(t);
	mask = abs(t - t.') <= opts.theiler;							% library exclusion, includes self
	p = 1 + 2*E; [ia, ib] = find(triu(true(p)));
	iSym1 = sub2ind([p p], ia, ib); iSym2 = sub2ind([p p], ib, ia);
	iDiag = sub2ind([p p], 2:p, 2:p);								% ridge on slopes, not the intercept
	nPr = size(pairs,1); rho = NaN(nPr,nTh); IS = NaN(nT,nTh,nPr);
	AR = []; if wantAR, AR = NaN(nT,E,nTh,nPr,'single'); end
	for q = 1:nPr
		x = Xb(:,pairs(q,1)); y = Xb(:,pairs(q,2));
		Zy = zeros(nT,E); Zx = zeros(nT,E);
		for l = 0:E-1, Zy(:,l+1) = y(t-l); Zx(:,l+1) = x(t-l); end
		Z = [Zy Zx]; yt = y(t+h); A = [ones(nT,1) Z];
		D = pdist2(Z, Z); Dl = D; Dl(mask) = NaN; dbar = mean(Dl, 2, 'omitnan');
		P = A(:,ia) .* A(:,ib); Q = A .* yt;
		for k = 1:nTh
			W = exp(-th(k) * D ./ dbar); W(mask) = 0;
			Gv = W * P; bv = W * Q;
			G = zeros(p*p, nT); G(iSym1,:) = Gv.'; G(iSym2,:) = Gv.';
			lam = opts.ridge * sum(G(iDiag,:),1) / (p-1);
			G(iDiag,:) = G(iDiag,:) + lam;
			beta = pagemldivide(reshape(G,p,p,nT), reshape(bv.',p,1,nT));
			beta = reshape(beta, p, nT);
			pred = sum(A .* beta.', 2);
			c = corrcoef(pred, yt); rho(q,k) = c(1,2);
			if wantIS, IS(:,k,q) = sum(beta(2+E:1+2*E,:), 1).'; end
			if wantAR, AR(:,:,k,q) = single(beta(2:1+E,:).'); end	% target's own lags: y(t), y(t-1), ...
		end
	end
end

