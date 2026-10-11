function outTL = tl_harness( f )
	S = load(f); X = fillmissing(S.X,'linear','EndValues','nearest');
	opts = struct('bin',5,'E',3,'h',1,'theiler',8,'theta',[0 0.5 2 4 16],'ridge',1e-4);
	Xb = local_bin(X,5); outTL = local_both(Xb, [1 2], opts); outTL = rmfield(outTL,'t');
	save(strrep(f,'.mat','_tl.mat'), '-struct', 'outTL');
end
function Xb = local_bin( X, b )
	n = floor(size(X,1)/b)*b;
	Xb = squeeze(mean(reshape(X(1:n,:), b, n/b, size(X,2)), 1));
	Xb = zscore(Xb);
end
function O = local_both( Xb, pairs, opts )
	n = size(Xb,1); E = opts.E; h = opts.h; th = opts.theta; nTh = numel(th);
	t = (E:n-h).'; nT = numel(t);
	dt = abs(t - t.'); mask = dt <= opts.theiler;
	p = 1 + 2*E; [ia, ib] = find(triu(true(p)));
	S.iSym1 = sub2ind([p p], ia, ib); S.iSym2 = sub2ind([p p], ib, ia); S.iDiag = sub2ind([p p], 2:p, 2:p); S.p = p; S.ridge = opts.ridge; S.E = E;
	% effective neighbour count of the time kernel on a bandwidth grid (depends only on the query position)
	bg = exp(linspace(log(0.25), log(50*nT), 120)); Nt = zeros(nT, numel(bg));
	for g = 1:numel(bg), w = exp(-dt/bg(g)); w(mask) = 0; Nt(:,g) = sum(w,2).^2 ./ sum(w.^2,2); end
	Nt = cummax(Nt, 2); lNt = log(Nt); lbg = log(bg);
	nPr = size(pairs,1);
	[O.rhoS, O.rhoT, O.NeffS, O.NeffT, O.bT, O.spreadS, O.spreadT] = deal(NaN(nPr,nTh));
	[O.ISS, O.IST] = deal(NaN(nT,nTh,nPr)); O.t = t;
	for q = 1:nPr
		x = Xb(:,pairs(q,1)); y = Xb(:,pairs(q,2));
		Zy = zeros(nT,E); Zx = zeros(nT,E);
		for l = 0:E-1, Zy(:,l+1) = y(t-l); Zx(:,l+1) = x(t-l); end
		Z = [Zy Zx]; yt = y(t+h); A = [ones(nT,1) Z];
		D = pdist2(Z, Z); Dl = D; Dl(mask) = NaN; dbar = mean(Dl, 2, 'omitnan');
		P = A(:,ia) .* A(:,ib); Q = A .* yt;
		for k = 1:nTh
			Ws = exp(-th(k) * D ./ dbar); Ws(mask) = 0;
			Ns = sum(Ws,2).^2 ./ sum(Ws.^2,2);
			[O.rhoS(q,k), O.ISS(:,k,q)] = local_solve(Ws, P, Q, A, yt, S);
			O.NeffS(q,k) = mean(Ns); O.spreadS(q,k) = mean(sum(Ws.*dt,2) ./ sum(Ws,2));
			if th(k) == 0		% global map: identical for both kernels
				O.rhoT(q,k) = O.rhoS(q,k); O.IST(:,k,q) = O.ISS(:,k,q); O.NeffT(q,k) = O.NeffS(q,k); O.bT(q,k) = Inf; O.spreadT(q,k) = O.spreadS(q,k);
				continue
			end
			b = local_match(lNt, lbg, log(Ns));
			Wt = exp(-dt ./ b); Wt(mask) = 0;
			[O.rhoT(q,k), O.IST(:,k,q)] = local_solve(Wt, P, Q, A, yt, S);
			O.NeffT(q,k) = mean(sum(Wt,2).^2 ./ sum(Wt.^2,2)); O.bT(q,k) = median(b); O.spreadT(q,k) = mean(sum(Wt.*dt,2) ./ sum(Wt,2));
		end
	end
end

function b = local_match( lNt, lbg, lNs )
% per query: the bandwidth whose time-kernel Kish N equals the state kernel's (log-log interpolation)
	[nT, nb] = size(lNt);
	g = sum(lNt < lNs, 2); g = min(max(g, 1), nb-1);
	i0 = sub2ind([nT nb], (1:nT).', g); i1 = i0 + nT;
	fr = (lNs - lNt(i0)) ./ max(lNt(i1) - lNt(i0), eps); fr = min(max(fr, 0), 1);
	b = exp(lbg(g).' + fr .* (lbg(g+1).' - lbg(g).'));
end

function [rho, IS] = local_solve( W, P, Q, A, yt, S )
	p = S.p; nT = size(W,1); E = S.E;
	Gv = W * P; bv = W * Q;
	G = zeros(p*p, nT); G(S.iSym1,:) = Gv.'; G(S.iSym2,:) = Gv.';
	lam = S.ridge * sum(G(S.iDiag,:),1) / (p-1);
	G(S.iDiag,:) = G(S.iDiag,:) + lam;
	beta = reshape(pagemldivide(reshape(G,p,p,nT), reshape(bv.',p,1,nT)), p, nT);
	pred = sum(A .* beta.', 2);
	c = corrcoef(pred, yt); rho = c(1,2);
	IS = sum(beta(2+E:1+2*E,:), 1).';
end

