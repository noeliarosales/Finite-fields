#---------------------------------------------------Funciones auxiliares---------------------------------------------------------------------
def es_primo(p):
    if p < 2:
        return False
    for i in range(2, p): 
        if p % i == 0:  #(Devuelve el resto, es decir, dice si p es  divisible por i (resto 0))
            return False
    return True

def factorizar_entero(n):
    fact=[]
    p=2
    while n>1:
        r=0
        while n%p==0:
            n//=p  #(Le da a n el valor del cociente entero de n entre p (n=n//p))
            r +=1  #(r es el numero de veces por el que se puede dividir p)
        if r !=0:  #(Si r es distinto de 0)
            fact.append((p,r))
        p +=1
    return fact

import random
import re 
import itertools
#-----------------------------------------------------------{F_p}------------------------------------------------------------------------------
class cuerpo_fp:  # construye el cuerpo de p elementos Fp = Z/pZ
   
    def __init__(self,p):
        if p<=1 or not es_primo(p):
            raise ValueError('p no es primo>=2')
        self.p=p
            
    def cero(self):             # devuelve el elemento 0
        return 0

    def uno(self):              # devuelve el elemento 1
        return 1
        
    def elem_de_int(self,n):    # fabrica el elemento dado por la clase de n
        return n% self.p

    def elem_de_str(self,s):    # fabrica el elemento a partir de un string (parser)
        return int(s)% self.p
        
    def conv_a_int(self, a):   # devuelve un entero entre 0 y p-1
        return int(a) % self.p
        
    def conv_a_str(self, a):   # pretty-printer
        return str(a)

    def suma(self,a,b):        # a+b
        return (a+b)% self.p

    def inv_adit(self,a):      # -a
        return (-a)%self.p

    def mult(self,a,b):        # a*b
        return (a*b) % self.p

    def pot(self,a,k):         # a^k (k entero)
        if k<0:
            a=self.inv_adit(a)
            k=-k
        if k==0:
            return self.uno() #hay que poner los paréntesis () para indicar que lo que quieres es lo que da one (el número)
        b= self.pot(a,k//2) # // indica división entera
        b=self.mult(b,b)
        if k%2==1: #k es impar
            b=self.mult(b,a)
        return b

    def inv_mult(self,a):     # a^(-1)
        if a==0:
            raise ValueError('0 no tiene inverso multiplicativo')
        return self.pot(a,self.p-2)
        
    def es_cero(self,a):      # a == 0
        return a==0    #Te da directamente: True or  False

    def es_uno(self, a):      # a == 1
        return a==1

    def es_igual(self, a, b): # a == b
        return a==b

    def aleatorio(self):     # fabrica un elemento aleatorio con prob uniforme
        return random.randint(0, self.p - 1)
        
    def tabla_suma(self):     # devuelve la matriz de pxp (lista de listas) de la suma        
        n=self.p #dimension de la matriz
        matriz = [[0 for _ in range(self.p)] for _ in range(self.p)]
        for i in range(n):
            for j in range(i,n):
                a=self.suma(i,j)
                matriz[i][j]=a
                matriz[j][i]=a
        return matriz

    def tabla_mult(self):     # devuelve la matriz de pxp (lista de listas) de la mult 
        matriz = [[0 for _ in range(self.p)] for _ in range(self.p)]
        for i in range(self.p):
            for j in range(i,self.p):
                a=self.mult(i,j)
                matriz[i][j]=a
                matriz[j][i]=a
        return matriz

    def tabla_inv_adit(self): # devuelve una lista de long p con los inv_adit
        return [self.inv_adit(x) for x in range(self.p)]

    def tabla_inv_mult(self): # devuelve una lista de long p con los inv_mult (en el índice 0 pone un '*')
        return['*']+[self.inv_mult(x) for x in range(1,self.p)]

    def cuadrado_latino(self, a): # cuadrado latino a*i+j (con a != 0)
        if a == 0:
            raise ValueError("a no puede ser 0")
        matriz = [[0 for _ in range(self.p)] for _ in range(self.p)]
        for i in range(self.p):          # filas
            for j in range(self.p):      # columnas
                matriz[i][j] = (a * i + j) % self.p  # cálculo de x = a*i + j mod p
        return matriz

    
#------------------------------------------------------------{F_p[x]}------------------------------------------------------------------------
class anillo_fp_x:
    
    def __init__(self,fp,var='x'): # construye el anillo Fp[var]
        self.fp=fp
        self.var=var
        
    def cero(self):               # 0
        return tuple()  #El polinimio vacío se interpreta como 0 (se puede operar con otra tupla, tenga la longitud que tenga)

    def uno(self):               # 1
        return (self.fp.uno(),) #Tupla con un solo elemento

    def reducir(self,a):        #Para que la representación de cada polinomio sea única, elimina los ceros de la derecha
        na=len(a)
        while na>0 and self.fp.es_cero(a[na-1]): # mientras el último coeficiente sea 0
            na-=1 # quitar ese coeficiente
        return a[:na]
        
    def elem_de_tuple(self,a):   # fabrica un polinomio a partir de la tupla (a0, a1, ...) (quita ceros y hace módulo)
        return self.reducir(tuple(self.fp.elem_de_int(n) for n in a))  
        
    def elem_de_int(self,a):    # fabrica un polinomio a partir de los dígitos de a en base p
        coef=[]
        while a>0:
            coef.append(a % self.fp.p) #el resto de la divison entre a y p
            a//=self.fp.p              #división entera
        return tuple(coef)

    def elem_de_str(self, p):    # fabrica un polinomio a partir de un string (parser)
        mod = self.fp.p       
        s = p.replace(' ', '')  
        term_pattern = re.compile(r'([+-]?\d*)(?:\*?x(?:\^(\d+))?)?')
        coef_dict = {}
        for match in term_pattern.finditer(s):
            if not match.group(0):
                continue
            coef_str, exp_str = match.groups()
            coef = int(coef_str) if coef_str not in ('', '+', '-') else (1 if coef_str in ('', '+') else -1)
            exp = int(exp_str) if exp_str else (1 if 'x' in match.group(0) else 0)
            coef_dict[exp] = (coef % mod)
                
        max_deg = max(coef_dict.keys(), default=-1)
        coef_tuple = tuple(coef_dict.get(i, 0) for i in range(max_deg + 1))
        return self.reducir(coef_tuple)


    def conv_a_tuple(self, a):      # devuelve la tupla de coeficientes
        return tuple(a)
        
    def conv_a_int(self, a):        # devuelve el entero correspondiente al polinomio
        b=0
        for i in range(len(a)):
            b+=a[i]*self.fp.p**i
        return b
        
    def conv_a_str(self, a):       # pretty-printer
        if not a:  # polinomio vacío = 0
            return "0"
        terminos = []
        for exp in range(len(a) - 1, -1, -1):  # recorrer desde grado mayor a menor
            coef = a[exp] % self.fp.p
            if coef == 0:
                continue
            if exp == 0:  # término independiente
                terminos.append(f"{coef}")
            elif exp == 1:  # término lineal
                if coef == 1:
                    terminos.append("x")
                else:
                    terminos.append(f"{coef}x")
            else:  # término de grado >= 2
                if coef == 1:
                    terminos.append(f"x^{exp}")
                else:
                    terminos.append(f"{coef}x^{exp}")
    
        return " + ".join(terminos) if terminos else "0"
        
    def suma(self,a,b):          # a+b
        na=len(a)
        nb=len(b)
        if na<nb:
            a,b,na,nb=b,a,nb,na
        c=tuple(self.fp.suma(a[i],b[i]) for i in range(nb))+a[nb:]
        return self.reducir(c)

    def inv_adit(self,a):        # -a
        return self.reducir(tuple(self.fp.inv_adit(x) for x in a))
        
    def mult(self,a,b):          # a*b
        na=len(a)
        nb=len(b)
        if na==0 or nb==0:
            return self.cero()
        c=[self.fp.cero()]*(na+nb-1) #Debe ser una lista ya que las tuplas son inmutables
        for i in range(na):
            for j in range(nb):
                c[i+j]=self.fp.suma(c[i+j],self.fp.mult(a[i],b[j]))
        return tuple(c)

    def mult_por_escalar(self,a,e): # a*e (con e en Z/pZ)
        na=len(a)
        return tuple(self.fp.mult(a[i],e) for i in range(na))

    def divmod(self,a,b):           # devuelve q,r tales que a=bq+r y deg(r)<deg(b)
        na=len(a)
        nb=len(b)
        if nb==0:
            raise ValueError('No se puede dividir por 0')
        if na<nb:
            return self.cero(), a
        q=[]                        #va guardando los coeficietes del cociente
        r=[x for x in a]
        y=self.fp.inv_mult(b[nb-1])
        for i in range(na-nb,-1,-1):#recorre desde na-nb con fin -1(antes de llegar a -1) y paso -1
            z=self.fp.mult(y,r[i+nb-1])
            q.append(z)
            z=self.fp.inv_adit(z)
            for j in range(nb-1):
                r[i+j]=self.fp.suma(r[i+j],self.fp.mult(z,b[j]))
            r[i+nb-1]=self.fp.cero()
        q.reverse()
        return tuple(q),self.reducir(tuple(r[:nb]))

    def div(self, a, b):           # q
        q, _ = self.divmod(a, b)
        return q
        
    def mod(self, a, b):           # r
        _,r = self.divmod(a, b)
        return r

    def grado(self,a):           # deg(a)
        return len(a)-1

    def gcd(self, a, b):           # devuelve g = gcd(a,b) mónico
        a = self.reducir(a)
        b = self.reducir(b)
        while b:                   #hasta que el polinomio b no sea 0, sigue iterando
            _, r = self.divmod(a, b)
            a, b = b, r
        if not a:  # gcd(0,0) = 0
            return self.cero()
        cp = a[-1]  # el último coeficiente de la tupla(coef principal)
        inv_cp = self.fp.inv_mult(cp)
        return self.mult_por_escalar(a, inv_cp)


            
    def gcd_ext(self, a, b):       # devuelve g,x,y tales que g=ax+by, g=gcd(a,b) mónico
        a = self.reducir(a)
        b = self.reducir(b)

        # Inicialización: combinaciones lineales
        u0, v0 = self.uno(), self.cero()  # para 'a'
        u1, v1 = self.cero(), self.uno()  # para 'b'

            # Inicialización de la secuencia
        r_minus2, r_minus1 = a, b
        alpha_minus2, alpha_minus1 = self.uno(), self.cero()
        beta_minus2, beta_minus1 = self.cero(), self.uno()
    
        while not self.es_cero(r_minus1):
            q, r = self.divmod(r_minus2, r_minus1)
    
            # Recurrencia
            r_next = r_minus2
            r_next = self.suma(r_next, self.inv_adit(self.mult(q, r_minus1)))
    
            alpha_next = self.suma(alpha_minus2, self.inv_adit(self.mult(q, alpha_minus1)))
            beta_next  = self.suma(beta_minus2, self.inv_adit(self.mult(q, beta_minus1)))
    
            # Avanzar
            r_minus2, r_minus1 = r_minus1, r
            alpha_minus2, alpha_minus1 = alpha_minus1, alpha_next
            beta_minus2, beta_minus1 = beta_minus1, beta_next
    
        # r_minus2 es el gcd, hacerlo mónico
        if self.es_cero(r_minus2):
            return self.cero(), self.cero(), self.cero()
    
        lc = r_minus2[-1]
        inv_lc = self.fp.inv_mult(lc)
    
        g = self.mult_por_escalar(r_minus2, inv_lc)
        alpha = self.mult_por_escalar(alpha_minus2, inv_lc)
        beta  = self.mult_por_escalar(beta_minus2, inv_lc)
    
        return g, alpha, beta
    
    def inv_mod(self, a, b):       # devuelve x tal que ax = 1 mod b
        g,x,y=self.gcd_ext(a,b)
        if g != self.uno():
            raise ValueError('No es invertible.')
        return x

    def pot_mod(self,a,k,b):
        if k<0:
            a=self.inv_mod(a,b)
            k=-k
        if k==0:
            return self.uno()
        elif k%2==0:
            x=self.pot_mod(a,k//2,b)
            x=self.mult(x,x)
            return self.mod(x,b)
        else:
            x=self.pot_mod(a,k-1,b)
            x=self.mult(x,a)
            return self.mod(x,b)
            
    def es_cero(self,a):      # a == 0
        return a==self.cero()

    def es_uno(self, a):      # a == 1
        return a==self.uno()

    def es_igual(self, a, b): # a == b
        a = self.reducir(a)
        b = self.reducir(b)
        if len(a) != len(b):
            return False
        return all(self.fp.es_igual(c1, c2) for c1, c2 in zip(a, b))

    def derivar(self,a):
        df=[0]*(len(a)-1)
        for i in range(0,len(df)):
            df[i]=self.fp.mult(a[i+1],i+1)
        return self.reducir(tuple(df))

    def monico(self,a):
        cp=a[-1]
        return self.mult_por_escalar(a,self.fp.inv_mult(cp))

    
#------------------------------------------------------------Fq={F_p[x]/<g(x)>}------------------------------------------------------------------------
class cuerpo_fq:
    def __init__(self,fp,g,var='a'):  # construye el cuerpo Fp[var]/<g(var)>  # g es objeto fabricado por fp
        self.fp=fp
        self.g=g
        self.var=var
        self.fpx=anillo_fp_x(fp,var='a')
        
    def cero(self):                   # 0
        n=self.fpx.grado(self.g)
        return tuple([0]*n)

    def uno(self):                    #1
        n=self.fpx.grado(self.g)
        return tuple([1]+[0]*(n-1))

    def elem_de_tuple(self, a):    # fabrica elemento a partir de tupla de coeficientes
        n=self.fpx.grado(self.g)
        a=self.fpx.elem_de_tuple(a)
        na=len(a)
        if na<=n:
            b=tuple(list(a)+[0]*(n-na))
        elif na>n:
            b=self.fpx.mod(a,self.g)
            if len(b)<n:
                b=tuple(list(b)+[0]*(n-len(b)))
        return tuple(b)
    
    def elem_de_int(self, a): # fabrica elemento a partir de entero
        return self.elem_de_tuple(self.fpx.elem_de_int(a))

    def elem_de_str(self, poly):
        mod = self.fp.p
        s = poly.replace(' ', '')  # quitar espacios
        term_pattern = re.compile(r'([+-]?\d*)(?:\*?x(?:\^(\d+))?)?')
        coef_dict = {}
    
        for match in term_pattern.finditer(s):
            if not match.group(0):
                continue
            coef_str, exp_str = match.groups()
            # coeficiente
            if coef_str in ('', '+'):
                coef = 1
            elif coef_str == '-':
                coef = -1
            else:
                coef = int(coef_str)
            # exponente
            if exp_str is not None:
                exp = int(exp_str)
            elif 'x' in match.group(0):
                exp = 1
            else:
                exp = 0
            coef_dict[exp] = coef % mod
    
        max_deg = max(coef_dict.keys(), default=-1)
        coef_tuple = tuple(coef_dict.get(i, 0) for i in range(max_deg + 1))
    
        # Reducir módulo el polinomio irreducible g
        _, c = self.fpx.divmod(coef_tuple, self.g)
        return tuple(c)
        
    def conv_a_tuple(self, a):     # devuelve tupla de coeficientes sin ceros "extra"        
        na=len(a)
        n=self.fpx.grado(self.g)
        if na<=n:
            return tuple(self.fpx.reducir(a))
        elif na>n:
            return ValueError('Los elementos de Fq son tuplas de n=deg(g) elementos.')
        return tuple(a)
        
    def conv_a_int(self, a):       # devuelve el entero correspondiente
        b=0
        for i in range(len(a)):
            b+=a[i]*self.fpx.fp.p**i
        return b
        
    def conv_a_str(self,a):
        partes = []
        for i, coef in enumerate(a):
            if coef == 0:
                continue  # ignorar coeficientes cero
            exp = i
            signo = '+' if coef > 0 else '-'
            coef_abs = abs(coef)
    
            # determinar la representación del término
            if exp == 0:
                termino = f"{coef_abs}"
            elif exp == 1:
                termino = "x" if coef_abs == 1 else f"{coef_abs}*x"
            else:
                termino = f"x^{exp}" if coef_abs == 1 else f"{coef_abs}*x^{exp}"
    
            partes.append(f"{signo} {termino}")
    
        if not partes:
            return '0'
    
        # unir partes y quitar signo inicial si es '+'
        s = partes[0]
        if s.startswith('+ '):
            s = s[2:]
        elif s.startswith('- '):
            s = '-' + s[2:]
    
        for p in partes[1:]:
            s += ' ' + p
    
        return s
    
    def suma(self, a, b):           # a+b
        c=self.fpx.suma(a,b)
        _,c=self.fpx.divmod(c,self.g)
        return self.elem_de_tuple(c)

    def inv_adit(self,a):
        c=self.fpx.inv_adit(a)
        return self.elem_de_tuple(c)

    def mult(self,a,b):             # a*b
        c=self.fpx.mult(a,b)
        _,c=self.fpx.divmod(c,self.g)
        return self.elem_de_tuple(c)

    def pot(self,a,k):             # a^k (k entero)
        c=self.fpx.pot_mod(a,k,self.g)
        return self.elem_de_tuple(c)

    def inv_mult(self,a):          # a^(-1)
        n=self.fpx.grado(self.g)
        q=pow(self.fp.p,n)
        b=self.pot(a,q-2)
        return b

    def es_cero(self,a):             # a == 0
        return a==self.cero()

    def es_uno(self,a):             # a == 1
        return a==self.uno()

    def es_igual(self,a,b):        # a == b
        return a==b

    def aleatorio(self):           # devuelve un elemento aleatorio con prob uniforme
        n=self.fpx.grado(self.g)
        alea=[0]*n
        for i in range(0,n):
            alea[i]=self.fpx.fp.aleatorio()
        return tuple(alea)

    def elementos(self):
        n=self.fpx.grado(self.g)
        tuplas=list(itertools.product(range(self.fp.p),repeat=n))
        tuplas=[tuple(t[::-1]) for t in tuplas]
        return tuplas
    
    def tabla_suma(self):            # matriz de qxq correspondiente a la suma (con la notación int)
        n=self.fpx.grado(self.g)
        q=pow(self.fp.p,n) #dimension de la matriz
        matriz = [[0 for _ in range(q)] for _ in range(q)]
        a=self.elementos()
        for i in range(q):
            for j in range(i,q):
                b=self.suma(a[i],a[j])
                c=self.conv_a_int(b)
                matriz[i][j]=c
                matriz[j][i]=c
        return matriz

    def tabla_mult(self):             # matriz de qxq correspondiente a la mult (con la notación int)
        n=self.fpx.grado(self.g)
        q=pow(self.fp.p,n) #dimension de la matriz
        matriz = [[0 for _ in range(q)] for _ in range(q)]
        a=self.elementos()
        for i in range(q):
            for j in range(i,q):
                b=self.mult(a[i],a[j])
                c=self.conv_a_int(b)
                matriz[i][j]=c
                matriz[j][i]=c
        return matriz

    def tabla_inv_adit(self):          # lista de inv_adit (con la notación int)
        n=self.fpx.grado(self.g)
        q=pow(self.fp.p,n) #longitud de la lista
        a=self.elementos()
        return [self.conv_a_int(self.inv_adit(x)) for x in a]

    def tabla_inv_mult(self):          # lista de inv_mult (con la notación int)
        n = self.fpx.grado(self.g)
        q = pow(self.fp.p, n)
        a = self.elementos()
        tabla = [self.conv_a_int(self.inv_mult(x)) for x in a]
        return ["*"] + tabla[1:]

    def cuadrado_latino(self, a):  # cuadrado latino para a != 0 (con notación int)
        if a==self.cero():
            return ValueError('a no puede ser cero.')
        n=self.fpx.grado(self.g)
        q=pow(self.fp.p,n)
        matriz = [[0 for _ in range(q)] for _ in range(q)]
        elem=self.elementos()
        for i in range(q):
            for j in range(q):
                suma = self.suma(self.mult(a, elem[i]), elem[j])
                matriz[i][j] = self.conv_a_int(suma)
        return matriz


#------------------------------------------------------------Fq[x]={F_p[x]/<g(x)>}[x]------------------------------------------------------------------------
class anillo_fq_x:
    def __init__(self, fq, var='x'): # Fq[var], var debe ser distinta que la de fq
        self.fq=fq
        self.var=var
    
    def cero(self):               # 0
        return tuple([])  #El polinimio vacío se interpreta como 0 (se puede operar con otra tupla, tenga la longitud que tenga)

    def uno(self):               # 1
        return (self.fq.uno(),) #Tupla con un solo elemento

    def elem_de_tuple(self,a):
        na=len(a)
        b=[0]*na
        for i in range(na):
            b[i]=self.fq.elem_de_tuple(a[i])
        return tuple(b)

    def elem_de_int(self, a):
        p=self.fq.fp.p
        n=self.fq.fpx.grado(self.fq.g)
        q=pow(p,n)
        elem_fq=self.fq.elementos()
        if a==0:
            return self.cero()
        tupla=[]
        resul=[]
        while a>0:
            tupla.append(a%q)
            a//=q
        for i in range(len(tupla)):
            ind=tupla[i]
            resul.append(elem_fq[ind])
        return tuple(resul)
        
    def elem_de_str(self, cadena):
        patron = r'\((.*?)\)'
        coefs_texto = re.findall(patron, cadena)
        s=[]
        for i in range(0,len(coefs_texto)):
            tupla=coefs_texto[i]
            s.append(self.fq.elem_de_str(tupla))
        return tuple(s)

    
    def conv_a_tuple(self, a):
        return tuple(a)
        
    def conv_a_int(self, a):
        p=self.fq.fp.p
        n=self.fq.fpx.grado(self.fq.g)
        q=pow(p,n)
        elem=self.fq.elementos()
        a1=list(a)
        num=0
        for i in range(0,len(a1)):
            indice=elem.index(a[i])
            num=num+indice*pow(q,i)
        return num
        
    def conv_a_str(self, a):
        partes = []
        for i, coef in enumerate(a):
            if coef == self.fq.cero(): 
                continue
    
            # Convertir el coeficiente (polinomio en x) a string
            coef_str = self.fq.conv_a_str(coef)
    
            # Encerrar entre paréntesis si tiene más de un término
            if '+' in coef_str or '-' in coef_str[1:]:
                coef_str = f"({coef_str})"
    
            # Variable principal: y
            exp_str = '' if i == 0 else ('y' if i == 1 else f"y^{i}")
    
            # Construir el término según el exponente
            if i == 0:
                term = coef_str
            elif coef_str in ('(1)', '1'):
                term = exp_str
            elif coef_str in ('(-1)', '-1'):
                term = f"-{exp_str}"
            else:
                term = f"{coef_str}*{exp_str}"
    
            partes.append(term)
    
        if not partes:
            return '0'
    
        # Unir con signos + y -
        resultado = partes[0]
        for t in partes[1:]:
            if t.startswith('-'):
                resultado += ' - ' + t[1:]
            else:
                resultado += ' + ' + t
        return resultado

    def reducir(self,a):
        na=len(a) #recibe tuplas de tuplas.
        while na>0 and a[na-1]==self.fq.cero():
            na -= 1
        return a[:na]

    def suma(self,a,b):
        cero = self.fq.cero()   # elemento neutro de F_q
        m = len(a)
        n = len(b)
        L = max(m, n)
        resultado = []

        a_ext = list(a) + [cero] * (L - m)
        b_ext = list(b) + [cero] * (L - n)
        # Sumar dentro de Fq
        
        resultado = [self.fq.suma(a, b) for a, b in zip(a_ext, b_ext)]
        return tuple(self.reducir(resultado))

    def inv_adit(self,a):
        a1=list(a)
        inv=[]
        for i in range(0,len(a1)):
            inv.append(self.fq.inv_adit(a1[i]))
        return self.reducir(tuple(inv))

    def mult(self, a, b):
        a1=list(a)
        b1=list(b)
        c=[self.fq.cero()]*(len(a1)+len(b1)-1)
        for i in range(0,len(a1)):
            for j in range(0,len(b1)):
                c[i+j]=self.fq.suma(c[i+j],self.fq.mult(a1[i],b1[j]))
        return self.reducir(tuple(c))
        
    def mult_por_escalar(self, a, e):
        a1=list(a)
        c=[self.fq.cero()]*len(a1)
        for i in range(0,len(a1)):
            c[i]=self.fq.mult(a1[i],e)
        return self.reducir(tuple(c))

    def divmod(self, a, b):
        a1=list(self.reducir(a))
        b1=list(self.reducir(b))
        na=len(a1)
        nb=len(b1)
        if nb==0:
            raise ValueError('El divisor es cero.')
        if na<nb:
            return self.cero(), a
        q=[]
        r=a1
        y=self.fq.inv_mult(b1[nb-1])
        for i in range(na-nb,-1,-1):
            z=self.fq.mult(y,r[i+nb-1])
            q.append(z)
            z=self.fq.inv_adit(z)
            for j in range(nb-1):
                r[i+j]=self.fq.suma(r[i+j],self.fq.mult(z,b1[j]))
            r[i+nb-1]=self.fq.cero()
        q.reverse()
        return tuple(q), self.reducir(tuple(r[:nb]))
        
    def div(self, a, b):
        return self.divmod(a,b)[0]
    
    def mod(self, a, b):
        return self.divmod(a,b)[1]
    
    def grado(self, a):
        a=self.reducir(a)
        return len(a)-1
    
    def gcd(self, a, b):
        a1=list(a)
        b1=list(b)
        na=len(a1)
        nb=len(b1)
        if na==0 and nb==0:
            raise ValueError('No existe el gcd de 0 y 0.')
        elif na==0:
            return self.mult_por_escalar(b,self.fq.inv_mult(b1[nb-1]))
        elif nb==0:
            return self.mult_por_escalar(a,self.fq.inv_mult(a1[na-1]))
        while b != self.cero():
            q, r = self.divmod(a,b)
            a = b
            b = r
        a2=list(a)
        coef=a2[len(a2)-1]
        return self.mult_por_escalar(a,self.fq.inv_mult(coef))

    def gcd_ext(self, a, b):
        a1=list(a)
        b1=list(b)
        na=len(a1)
        nb=len(b1)
        if na==0 and nb==0:
            raise ValueError('No existe el gcd de 0 y 0.')
        elif na==0:
            y=self.fq.inv_mult(b1[nb-1])
            return self.mult_por_escalar(b,y), self.cero(), tuple([y])
        elif nb==0:
            x=self.fq.inv_mult(a1[na-1])
            return self.mult_por_escalar(a,x), tuple([x]), self.cero()
        r0=a
        r1=b
        s0=self.uno()
        s1=self.cero()
        t0=self.cero()
        t1=self.uno()
        while r1 != self.cero():
            q,r=self.divmod(r0,r1)
            r0=r1
            r1=r
            s0,s1 = s1,self.suma(s0,self.inv_adit(self.mult(q,s1)))
            t0,t1 = t1,self.suma(t0,self.inv_adit(self.mult(q,t1)))
        r2=list(r0)
        coef=r2[-1]
        inv_coef=self.fq.inv_mult(coef)
        return self.mult_por_escalar(r0,inv_coef), self.mult_por_escalar(s0,inv_coef), self.mult_por_escalar(t0,inv_coef)
        
    def inv_mod(self, a, b):
        g,x,y=self.gcd_ext(a,b)
        if g != self.uno():
            raise ValueError('No es invertible.')
        return x
        
    def pot_mod(self, a, k, b):
        if k<0:
            a=self.inv_mod(a,b)
            k=-k
        if k==0:
            return self.uno()
        elif k%2==0:
            x=self.pot_mod(a,k//2,b)
            x=self.mult(x,x)
            return self.divmod(x,b)[1]
        else:
            x=self.pot_mod(a,k-1,b)
            x=self.mult(x,a)
            return self.divmod(x,b)[1]
            
    def es_cero(self, a):
        return a==self.cero()
        
    def es_uno(self, a):
        return a==self.uno()
        
    def es_igual(self, a, b):
        return a==b
        
    def derivar(self,f):
        f1=list(f)
        df=[0]*(len(f1)-1)
        for i in range(1,len(f1)):
            df[i-1]=self.fq.fpx.mult_por_escalar(f1[i],i)
            df[i-1]=self.fq.elem_de_tuple(df[i-1])
        return self.reducir(tuple(df))
        
    def monico(self,f):
        f1=list(f)
        coef=f1[-1]
        return self.mult_por_escalar(f,self.fq.inv_mult(coef))
        
