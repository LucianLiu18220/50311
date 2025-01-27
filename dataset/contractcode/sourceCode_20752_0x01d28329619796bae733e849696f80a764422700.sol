/**
 *Submitted for verification at Etherscan.io on 2018-01-27
*/

pragma solidity ^0.4.15;

library SafeMath {

    function mul(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a * b;
        assert(a == 0 || c / a == b);
        return c;
    }

    function div(uint256 a, uint256 b) internal pure returns (uint256) {
        // assert(b > 0); // Solidity automatically throws when dividing by 0
        uint256 c = a / b;
        // assert(a == b * c + a % b); // There is no case in which this doesn't hold
        return c;
    }

    function sub(uint256 a, uint256 b) internal pure returns (uint256) {
        assert(b <= a);
        return a - b;
    }

    function add(uint256 a, uint256 b) internal pure returns (uint256) {
        uint256 c = a + b;
        assert(c >= a);
        return c;
    }
}

contract ApproveAndCallReceiver {
    function receiveApproval(
        address _from, 
        uint256 _amount, 
        address _token, 
        bytes _data
    ) public;
}

//normal contract. already compiled as bin
contract Controlled {
    /// @notice The address of the controller is the only address that can call
    ///  a function with this modifier
    modifier onlyController { 
        require(msg.sender == controller); 
        _; 
    }

    //block for check//bool private initialed = false;
    address public controller;

    function Controlled() public {
      //block for check//require(!initialed);
      controller = msg.sender;
      //block for check//initialed = true;
    }

    /// @notice Changes the controller of the contract
    /// @param _newController The new controller of the contract
    function changeController(address _newController) onlyController public {
        controller = _newController;
    }
}

contract ERC20Token {
    /* This is a slight change to the ERC20 base standard.
      function totalSupply() constant returns (uint256 supply);
      is replaced with:
      uint256 public totalSupply;
      This automatically creates a getter function for the totalSupply.
      This is moved to the base contract since public getter functions are not
      currently recognised as an implementation of the matching abstract
      function by the compiler.
    */
    /// total amount of tokens
    uint256 public totalSupply;
    //function totalSupply() public constant returns (uint256 balance);

    /// @param _owner The address from which the balance will be retrieved
    /// @return The balance
    mapping (address => uint256) public balanceOf;
    //function balanceOf(address _owner) public constant returns (uint256 balance);

    /// @notice send `_value` token to `_to` from `msg.sender`
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transfer(address _to, uint256 _value) public returns (bool success);

    /// @notice send `_value` token to `_to` from `_from` on the condition it is approved by `_from`
    /// @param _from The address of the sender
    /// @param _to The address of the recipient
    /// @param _value The amount of token to be transferred
    /// @return Whether the transfer was successful or not
    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success);

    /// @notice `msg.sender` approves `_spender` to spend `_value` tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @param _value The amount of tokens to be approved for transfer
    /// @return Whether the approval was successful or not
    function approve(address _spender, uint256 _value) public returns (bool success);

    /// @param _owner The address of the account owning tokens
    /// @param _spender The address of the account able to transfer the tokens
    /// @return Amount of remaining tokens allowed to spent
    mapping (address => mapping (address => uint256)) public allowance;
    //function allowance(address _owner, address _spender) public constant returns (uint256 remaining);

    event Transfer(address indexed _from, address indexed _to, uint256 _value);
    event Approval(address indexed _owner, address indexed _spender, uint256 _value);
}

//abstract contract. used for interface
contract TokenController {
    /// @notice Called when `_owner` sends ether to the MiniMe Token contract
    /// @param _owner The address that sent the ether to create tokens
    /// @return True if the ether is accepted, false if it throws
    function proxyPayment(address _owner) payable public returns(bool);

    /// @notice Notifies the controller about a token transfer allowing the
    ///  controller to react if desired
    /// @param _from The origin of the transfer
    /// @param _to The destination of the transfer
    /// @param _amount The amount of the transfer
    /// @return False if the controller does not authorize the transfer
    function onTransfer(address _from, address _to, uint _amount) public returns(bool);

    /// @notice Notifies the controller about an approval allowing the
    ///  controller to react if desired
    /// @param _owner The address that calls `approve()`
    /// @param _spender The spender in the `approve()` call
    /// @param _amount The amount in the `approve()` call
    /// @return False if the controller does not authorize the approval
    function onApprove(address _owner, address _spender, uint _amount) public returns(bool);
}


contract TokenI is ERC20Token, Controlled {

    string public name;                //The Token's name: e.g. DigixDAO Tokens
    uint8 public decimals;             //Number of decimals of the smallest unit
    string public symbol;              //An identifier: e.g. REP

///////////////////
// ERC20 Methods
///////////////////

    /// @notice `msg.sender` approves `_spender` to send `_amount` tokens on
    ///  its behalf, and then a function is triggered in the contract that is
    ///  being approved, `_spender`. This allows users to use their tokens to
    ///  interact with contracts in one function call instead of two
    /// @param _spender The address of the contract able to transfer the tokens
    /// @param _amount The amount of tokens to be approved for transfer
    /// @return True if the function call was successful
    function approveAndCall(
        address _spender,
        uint256 _amount,
        bytes _extraData
    ) public returns (bool success);

////////////////
// Generate and destroy tokens
////////////////

    /// @notice Generates `_amount` tokens that are assigned to `_owner`
    /// @param _owner The address that will be assigned the new tokens
    /// @param _amount The quantity of tokens generated
    /// @return True if the tokens are generated correctly
    function generateTokens(address _owner, uint _amount) public returns (bool);


    /// @notice Burns `_amount` tokens from `_owner`
    /// @param _owner The address that will lose the tokens
    /// @param _amount The quantity of tokens to burn
    /// @return True if the tokens are burned correctly
    function destroyTokens(address _owner, uint _amount) public returns (bool);

////////////////
// Enable tokens transfers
////////////////

    /// @notice Enables token holders to transfer their tokens freely if true
    /// @param _transfersEnabled True if transfers are allowed in the clone
    function enableTransfers(bool _transfersEnabled) public;

//////////
// Safety Methods
//////////

    /// @notice This method can be used by the controller to extract mistakenly
    ///  sent tokens to this contract.
    /// @param _token The address of the token contract that you want to recover
    ///  set to 0 in case you want to extract ether.
    function claimTokens(address _token) public;

////////////////
// Events
////////////////

    event ClaimedTokens(address indexed _token, address indexed _controller, uint _amount);
}

contract Token is TokenI {
    using SafeMath for uint256;

    address public owner;

    uint256 public maximumToken = 10 * 10**8 * 10**18; //总发行量1b

    struct FreezeInfo {
        address user;
        uint256 amount;
    }
    //Key1: step(募资阶段); Key2: user sequence(用户序列)
    mapping (uint8 => mapping (uint8 => FreezeInfo)) public freezeOf; //所有锁仓，key 使用序号向上增加，方便程序查询。
    mapping (uint8 => uint8) public lastFreezeSeq; //最后的 freezeOf 键值。key: step; value: sequence
    mapping (uint8 => uint8) internal unlockTime;

    bool public transfersEnabled;

    /* This generates a public event on the blockchain that will notify clients */
    //event Transfer(address indexed from, address indexed to, uint256 value);

    /* This notifies clients about the amount burnt */
    event Burn(address indexed from, uint256 value);
    
    /* This notifies clients about the amount frozen */
    event Freeze(address indexed from, uint256 value);
    
    /* This notifies clients about the amount unfrozen */
    event Unfreeze(address indexed from, uint256 value);

    /* Initializes contract with initial supply tokens to the creator of the contract */
    function Token(
        uint256 initialSupply,
        string tokenName,
        uint8 decimalUnits,
        string tokenSymbol,
        bool transfersEnable
        ) public {
        balanceOf[msg.sender] = initialSupply;
        totalSupply = initialSupply;
        name = tokenName;
        symbol = tokenSymbol;
        decimals = decimalUnits;
        transfersEnabled = transfersEnable;
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner);
        _;
    }

    modifier ownerOrUser(address user){
        require(msg.sender == owner || msg.sender == user);
        _;
    }

    modifier realUser(address user){
        if(user == 0x0){
            revert();
        }
        _;
    }

    modifier moreThanZero(uint256 _value){
        if (_value <= 0){
            revert();
        }
        _;
    }

    modifier moreOrEqualZero(uint256 _value){
        if(_value < 0){
            revert();
        }
        _;
    }

    /// @dev Internal function to determine if an address is a contract
    /// @param _addr The address being queried
    /// @return True if `_addr` is a contract
    function isContract(address _addr) constant internal returns(bool) {
        uint size;
        if (_addr == 0) {
            return false;
        }
        assembly {
            size := extcodesize(_addr)
        }
        return size>0;
    }

    /* Send coins */
    function transfer(address _to, uint256 _value) realUser(_to) moreThanZero(_value) public returns (bool) {
        require(balanceOf[msg.sender] > _value);           // Check if the sender has enough
        require(balanceOf[_to] + _value > balanceOf[_to]); // Check for overflows
        balanceOf[msg.sender] = balanceOf[msg.sender] - _value;                     // Subtract from the sender
        balanceOf[_to] = balanceOf[_to] + _value;                            // Add the same to the recipient
        Transfer(msg.sender, _to, _value);                   // Notify anyone listening that this transfer took place
        return true;
    }

    /* Allow another contract to spend some tokens in your behalf */
    function approve(address _spender, uint256 _value) moreThanZero(_value) public
        returns (bool success) {
        allowance[msg.sender][_spender] = _value;
        return true;
    }

    /**
     * @notice `msg.sender` approves `_spender` to send `_amount` tokens on
     *  its behalf, and then a function is triggered in the contract that is
     *  being approved, `_spender`. This allows users to use their tokens to
     *  interact with contracts in one function call instead of two
     * @param _spender The address of the contract able to transfer the tokens
     * @param _amount The amount of tokens to be approved for transfer
     * @return True if the function call was successful
     */
    function approveAndCall(address _spender, uint256 _amount, bytes _extraData) public returns (bool success) {
        require(approve(_spender, _amount));

        ApproveAndCallReceiver(_spender).receiveApproval(
            msg.sender,
            _amount,
            this,
            _extraData
        );

        return true;
    }

    /* A contract attempts to get the coins */
    function transferFrom(address _from, address _to, uint256 _value) realUser(_from) realUser(_to) moreThanZero(_value) public returns (bool success) {
        require(balanceOf[_from] > _value);                 // Check if the sender has enough
        require(balanceOf[_to] + _value > balanceOf[_to]);  // Check for overflows
        require(_value < allowance[_from][msg.sender]);     // Check allowance
        balanceOf[_from] = balanceOf[_from] - _value;                           // Subtract from the sender
        balanceOf[_to] = balanceOf[_to] + _value;                             // Add the same to the recipient
        allowance[_from][msg.sender] = allowance[_from][msg.sender] + _value;
        Transfer(_from, _to, _value);
        return true;
    }
    
    //只能自己或者 owner 才能冻结账户
    function freeze(address _user, uint256 _value, uint8 _step) moreThanZero(_value) onlyController public returns (bool success) {
        //info256("balanceOf[_user]", balanceOf[_user]);
        require(balanceOf[_user] >= _value);
        balanceOf[_user] = balanceOf[_user] - _value;
        freezeOf[_step][lastFreezeSeq[_step]] = FreezeInfo({user:_user, amount:_value});
        lastFreezeSeq[_step]++;
        Freeze(_user, _value);
        return true;
    }

    event info(string name, uint8 value);
    event info256(string name, uint256 value);
    
    //为用户解锁账户资金
    function unFreeze(uint8 _step) onlyController public returns (bool unlockOver) {
        //_end = length of freezeOf[_step]
        uint8 _end = lastFreezeSeq[_step];
        require(_end > 0);
        //info("_end", _end);
        unlockOver = (_end <= 49);
        uint8 _start = (_end > 49) ? _end-50 : 0;
        //info("_start", _start);
        for(; _end>_start; _end--){
            FreezeInfo storage fInfo = freezeOf[_step][_end-1];
            uint256 _amount = fInfo.amount;
            balanceOf[fInfo.user] += _amount;
            delete freezeOf[_step][_end-1];
            lastFreezeSeq[_step]--;
            Unfreeze(fInfo.user, _amount);
        }
    }
    
    //accept ether
    function() payable public {
        //屏蔽控制方的合约类型检查，以兼容发行方无控制合约的情况。
        require(isContract(controller));
        bool proxyPayment = TokenController(controller).proxyPayment.value(msg.value)(msg.sender);
        require(proxyPayment);
    }

////////////////
// Generate and destroy tokens
////////////////

    /// @notice Generates `_amount` tokens that are assigned to `_owner`
    /// @param _user The address that will be assigned the new tokens
    /// @param _amount The quantity of tokens generated
    /// @return True if the tokens are generated correctly
    function generateTokens(address _user, uint _amount) onlyController public returns (bool) {
        require(balanceOf[owner] >= _amount);
        balanceOf[_user] += _amount;
        balanceOf[owner] -= _amount;
        Transfer(0, _user, _amount);
        return true;
    }

    /// @notice Burns `_amount` tokens from `_owner`
    /// @param _user The address that will lose the tokens
    /// @param _amount The quantity of tokens to burn
    /// @return True if the tokens are burned correctly
    function destroyTokens(address _user, uint _amount) onlyController public returns (bool) {
        balanceOf[owner] += _amount;
        balanceOf[_user] -= _amount;
        Transfer(_user, 0, _amount);
        return true;
    }

    function changeOwner(address newOwner) onlyOwner public returns (bool) {
        balanceOf[newOwner] = balanceOf[owner];
        balanceOf[owner] = 0;
        owner = newOwner;
        return true;
    }

////////////////
// Enable tokens transfers
////////////////

    /// @notice Enables token holders to transfer their tokens freely if true
    /// @param _transfersEnabled True if transfers are allowed in the clone
    function enableTransfers(bool _transfersEnabled) onlyController public {
        transfersEnabled = _transfersEnabled;
    }

//////////
// Safety Methods
//////////

    /// @notice This method can be used by the controller to extract mistakenly
    ///  sent tokens to this contract.
    ///  set to 0 in case you want to extract ether.
    function claimTokens(address _token) onlyController public {
        if (_token == 0x0) {
            controller.transfer(this.balance);
            return;
        }

        Token token = Token(_token);
        uint balance = token.balanceOf(this);
        token.transfer(controller, balance);
        ClaimedTokens(_token, controller, balance);
    }
}

contract BaseTokenSale is TokenController, Controlled {

    using SafeMath for uint256;

    uint256 public startFundingTime;
    uint256 public endFundingTime;
    
    uint256 constant public maximumFunding = 1951 ether; //硬顶
    uint256 public maxFunding;  //最高投资额度
    uint256 public minFunding = 0.001 ether;  //最低起投额度
    uint256 public tokensPerEther = 41000;
    uint256 constant public maxGasPrice = 50000000000;
    uint256 constant oneDay = 86400;
    uint256 public totalCollected = 0;
    bool    public paused;
    Token public tokenContract;
    bool public finalized = false;
    bool public allowChange = true;
    bool private transfersEnabled = true;
    address private vaultAddress;

    bool private initialed = false;

    event Payment(address indexed _sender, uint256 _ethAmount, uint256 _tokenAmount);

    /**
     * @param _startFundingTime The UNIX time that the BaseTokenSale will be able to start receiving funds
     * @param _endFundingTime   The UNIX time that the BaseTokenSale will stop being able to receive funds
     * @param _vaultAddress     The address that will store the donated funds
     * @param _tokenAddress     Address of the token contract this contract controls
     */
    function BaseTokenSale(
        uint _startFundingTime, 
        uint _endFundingTime, 
        address _vaultAddress,
        address _tokenAddress
    ) public {
        require(_endFundingTime > now);
        require(_endFundingTime >= _startFundingTime);
        require(_vaultAddress != 0);
        require(_tokenAddress != 0);
        require(!initialed);

        startFundingTime = _startFundingTime;
        endFundingTime = _endFundingTime;
        vaultAddress = _vaultAddress;
        tokenContract = Token(_tokenAddress);
        paused = false;
        initialed = true;
    }


    function setTime(uint time1, uint time2) onlyController public {
        require(endFundingTime > now && startFundingTime < endFundingTime);
        startFundingTime = time1;
        endFundingTime = time2;
    }


    /**
     * @dev The fallback function is called when ether is sent to the contract, it simply calls `doPayment()` with the address that sent the ether as the `_owner`. Payable is a required solidity modifier for functions to receive ether, without this modifier functions will throw if ether is sent to them
     */
    function () payable notPaused public {
        doPayment(msg.sender);
    }

    /**
     * @notice `proxyPayment()` allows the caller to send ether to the BaseTokenSale and have the tokens created in an address of their choosing
     * @param _owner The address that will hold the newly created tokens
     */
    function proxyPayment(address _owner) payable notPaused public returns(bool success) {
        return doPayment(_owner);
    }

    /**
    * @notice Notifies the controller about a transfer, for this BaseTokenSale all transfers are allowed by default and no extra notifications are needed
    * @param _from The origin of the transfer
    * @param _to The destination of the transfer
    * @param _amount The amount of the transfer
    * @return False if the controller does not authorize the transfer
    */
    function onTransfer(address _from, address _to, uint _amount) public returns(bool success) {
        if ( _from == vaultAddress || transfersEnabled) {
            return true;
        }
        _to;
        _amount;
        return false;
    }

    /**
     * @notice Notifies the controller about an approval, for this BaseTokenSale all
     * approvals are allowed by default and no extra notifications are needed
     * @param _owner The address that calls `approve()`
     * @param _spender The spender in the `approve()` call
     * @param _amount The amount in the `approve()` call
     * @return False if the controller does not authorize the approval
     */
    function onApprove(address _owner, address _spender, uint _amount) public returns(bool success) {
        if ( _owner == vaultAddress ) {
            return true;
        }
        _spender;
        _amount;
        return false;
    }

    event info(string name, string msg);
    event info256(string name, uint256 value);

    /// @dev `doPayment()` is an internal function that sends the ether that this
    ///  contract receives to the `vault` and creates tokens in the address of the
    ///  `_owner` assuming the BaseTokenSale is still accepting funds
    /// @param _owner The address that will hold the newly created tokens
    function doPayment(address _owner) internal returns(bool success) {
        //info("step", "enter doPayment");
        require(msg.value >= minFunding);
        require(endFundingTime > now);

        // Track how much the BaseTokenSale has collected
        require(totalCollected < maximumFunding);
        totalCollected = totalCollected.add(msg.value);

        //Send the ether to the vault
        require(vaultAddress.send(msg.value));
        
        uint256 tokenValue = tokensPerEther.mul(msg.value);
        // Creates an equal amount of tokens as ether sent. The new tokens are created in the `_owner` address
        require(tokenContract.generateTokens(_owner, tokenValue));
        uint256 lock1 = tokenValue / 10;    //前5个月按照每月10%锁定
        uint256 lock2 = tokenValue / 5;     //最后一个月解锁20%
        require(tokenContract.freeze(_owner, lock1, 0)); //私募第一轮解锁
        tokenContract.freeze(_owner, lock1, 1); //私募第二轮解锁
        tokenContract.freeze(_owner, lock1, 2);
        tokenContract.freeze(_owner, lock1, 3);
        tokenContract.freeze(_owner, lock1, 4);
        tokenContract.freeze(_owner, lock2, 5);
        //require(tokenContract.freeze(_owner, lock3, 5)); //私募第三轮解锁
        Payment(_owner, msg.value, tokenValue);
        return true;
    }

    function changeTokenController(address _newController) onlyController public {
        tokenContract.changeController(_newController);
    }

    /**
     * 修改TNB兑换比率
     */
    function changeTokensPerEther(uint256 _newRate) onlyController public {
        require(allowChange);
        tokensPerEther = _newRate;
    }

    function changeFundingLimit(uint256 _min, uint256 _max) onlyController public {
        require(_min > 0 && _min <= _max);
        minFunding = _min;
        maxFunding = _max;
    }

    /**
     * 允许普通用户转账
     */
    function allowTransfersEnabled(bool _allow) onlyController public {
        transfersEnabled = _allow;
    }

    /// @dev Internal function to determine if an address is a contract
    /// @param _addr The address being queried
    /// @return True if `_addr` is a contract
    function isContract(address _addr) constant internal returns (bool) {
        if (_addr == 0) {
            return false;
        }
        uint256 size;
        assembly {
            size := extcodesize(_addr)
        }
        return (size > 0);
    }

    /// @notice `finalizeSale()` ends the BaseTokenSale. It will generate the platform and team tokens
    ///  and set the controller to the referral fee contract.
    /// @dev `finalizeSale()` can only be called after the end of the funding period or if the maximum amount is raised.
    function finalizeSale() onlyController public {
        require(now > endFundingTime || totalCollected >= maximumFunding);
        require(!finalized);

        //20000 TNB/ETH and 90 percent discount
        uint256 totalTokens = totalCollected * tokensPerEther * 10**18;
        if (!tokenContract.generateTokens(vaultAddress, totalTokens)) {
            revert();
        }

        finalized = true;
        allowChange = false;
    }

//////////
// Safety Methods
//////////

    /// @notice This method can be used by the controller to extract mistakenly
    ///  sent tokens to this contract.
    /// @param _token The address of the token contract that you want to recover
    ///  set to 0 in case you want to extract ether.
    function claimTokens(address _token) onlyController public {
        if (_token == 0x0) {
            controller.transfer(this.balance);
            return;
        }

        ERC20Token token = ERC20Token(_token);
        uint balance = token.balanceOf(this);
        token.transfer(controller, balance);
        ClaimedTokens(_token, controller, balance);
    }

    event ClaimedTokens(address indexed _token, address indexed _controller, uint _amount);

  /// @notice Pauses the contribution if there is any issue
    function pauseContribution() onlyController public {
        paused = true;
    }

    /// @notice Resumes the contribution
    function resumeContribution() onlyController public {
        paused = false;
    }

    modifier notPaused() {
        require(!paused);
        _;
    }
}